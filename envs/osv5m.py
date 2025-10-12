from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any
from pathlib import Path

import csv
from io import StringIO

import pandas as pd
from PIL import Image
import re
import unicodedata
from math import exp

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


def _resolve_eos_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
        if eos_id is not None:
            return eos_id
    if hasattr(tokenizer, "get_eos_token_id"):
        return tokenizer.get_eos_token_id()
    return None


@dataclass(frozen=True)
class OSV5MPrompt:
    """Observation returned by `OSV5MEnv.reset` for visual geolocation.

    Includes a natural-language `question` and either a PIL `image` or an
    `image_path` string so evaluation can construct a VLM chat prompt.
    """

    question: str
    image: object  # PIL Image
    image_path: str
    latitude: float
    longitude: float
    country: str
    region: str
    sub_region: str
    city: str
    prediction_mode: str
    reward_weights: Dict[str, float]
    reward_stage: str


@dataclass(frozen=True)
class OSV5MState(BaseState):
    dataset_idx: int
    image: object  # PIL Image
    latitude: float
    longitude: float
    country: str
    region: str
    sub_region: str
    city: str
    prediction_mode: str
    reward_weights: Dict[str, float]
    reward_stage: str

    def render(self) -> str:
        return "Looking at this street view image, predict the location (country, region, or coordinates)."


class OSV5MEnv(BaseEnv):
    """OpenStreetView-5M visual geolocation environment.

    Each task presents a streetview image and the agent must predict the geographic
    location. Supports multiple evaluation modes:
    - Country prediction (easiest)
    - Region prediction (medium)
    - City prediction (harder)
    - Coordinate prediction (hardest)

    Rewards based on prediction accuracy at the specified granularity.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "test",
        difficulty_schedule: Optional[Dict[int, Any]] = None,
        coord_tolerance_km: float = 25.0,  # legacy tolerance (still used for info)
        max_samples: int = 1000,  # limit dataset size for efficiency
        # Hierarchical reward weights (sum <= 1.0 recommended)
        country_weight: float = 0.2,
        region_weight: float = 0.3,
        city_weight: float = 0.5,
        # Enable distance-based shaping r_geo = exp(-d/geo_decay_km)
        use_geo_shaping: bool = True,
        geo_decay_km: float = 300.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 256  # More tokens for potentially complex location descriptions
        self.split = split
        self.coord_tolerance_km = coord_tolerance_km
        self.episode_count = 0
        self.max_samples = max_samples
        self.dataset = self._load_dataset()
        # Reward settings
        self.target_weights = {
            "country": float(country_weight),
            "region": float(region_weight),
            "city": float(city_weight),
        }
        self.use_geo_shaping = bool(use_geo_shaping)
        self.geo_decay_km = float(geo_decay_km)
        self.prompt_template = self._build_prompt_template()
        self.weight_schedule = self._prepare_weight_schedule(difficulty_schedule, self.target_weights)
        self._current_stage = "full"

        # Safety check for empty dataset
        if len(self.dataset) == 0:
            raise ValueError(
                f"No valid samples found for OSV5M {split} dataset. "
                f"Please check that images exist in the expected location. "
                f"Downloaded images should be in: ~/.cache/huggingface/datasets/osv5m/images/{split}/"
            )

        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _load_dataset(self) -> Sequence[dict]:
        """Load OSV5M dataset.

        Downloads the dataset if not already cached and loads metadata + images.
        """
        cache_dir = Path.home() / ".cache/huggingface/datasets/osv5m"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download metadata
        metadata_file = cache_dir / f"{self.split}.csv"
        if not metadata_file.exists():
            print(f"Downloading OSV5M {self.split} metadata...")
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="osv5m/osv5m",
                filename=f"{self.split}.csv",
                repo_type="dataset",
                local_dir=str(cache_dir),
            )

        # Load metadata
        df = pd.read_csv(metadata_file, dtype={"id": str})

        # Download image zips if needed (just download first batch for testing)
        images_dir = cache_dir / "images" / self.split
        if not images_dir.exists():
            print(f"Downloading OSV5M {self.split} images (this may take a while)...")
            from huggingface_hub import hf_hub_download

            images_dir.mkdir(parents=True, exist_ok=True)

            # For test set, download first zip file only for efficiency
            if self.split == "test":
                num_zips = 1  # Just first batch for testing
            else:
                num_zips = 2  # Limited for training too

            for i in range(num_zips):
                zip_name = str(i).zfill(2) + ".zip"
                zip_path = hf_hub_download(
                    repo_id="osv5m/osv5m",
                    filename=zip_name,
                    subfolder=f"images/{self.split}",
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )

                # Extract zip
                print(f"Extracting {zip_name}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(images_dir)

        # Build dataset entries - only include entries with existing images
        parsed = []
        checked_count = 0
        max_checks = min(len(df), 100000)  # Don't check more than 100k entries to avoid long waits

        print(f"Searching for available images in {images_dir}...")
        for _, row in df.iterrows():
            checked_count += 1
            if checked_count > max_checks:
                print(f"Checked {max_checks} entries, stopping search")
                break

            img_id = row["id"]
            # For test set, all images from 00.zip are in the 00 folder
            # regardless of their ID prefix
            if self.split == "test":
                # Try the 00 folder first (where 00.zip extracts to)
                img_path = images_dir / "00" / f"{img_id}.jpg"
            else:
                # For other splits, try the folder based on first 2 digits
                img_folder = img_id[:2] if len(img_id) >= 2 else "00"
                img_path = images_dir / img_folder / f"{img_id}.jpg"

            # Skip if image doesn't exist (we only downloaded partial dataset)
            if not img_path.exists():
                continue

            parsed.append(
                {
                    "id": img_id,
                    "image_path": str(img_path),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "country": str(row["country"]) if pd.notna(row["country"]) else "",
                    "region": str(row["region"]) if pd.notna(row["region"]) else "",
                    "sub_region": str(row["sub-region"]) if pd.notna(row["sub-region"]) else "",
                    "city": str(row["city"]) if pd.notna(row["city"]) else "",
                }
            )

            # Stop early if we have enough samples
            if len(parsed) >= self.max_samples:
                print(f"Found {len(parsed)} images (reached max_samples limit)")
                break

        if len(parsed) == 0:
            print(f"No images found after checking {checked_count} entries")

        print(f"Loaded {len(parsed)} OSV5M samples")
        return parsed

    def _build_prompt_template(self) -> str:
        """Return a unified structured prompt."""
        return """Guess the City, Region, Country, Latitude, Longitude of the image in this format."""

    def _prepare_weight_schedule(
        self,
        schedule: Optional[Dict[int, Any]],
        target_weights: Dict[str, float],
    ) -> Dict[int, Dict[str, Any]]:
        """Normalize reward-weight schedule; default to simple curriculum."""
        def weights_for_stage(stage: str) -> Dict[str, float]:
            stage = stage.lower()
            if stage == "country":
                return {"country": target_weights["country"], "region": 0.0, "city": 0.0}
            if stage == "region":
                return {"country": target_weights["country"], "region": target_weights["region"], "city": 0.0}
            if stage == "city":
                return {
                    "country": target_weights["country"],
                    "region": target_weights["region"],
                    "city": target_weights["city"],
                }
            if stage == "coords":
                return {
                    "country": target_weights["country"],
                    "region": target_weights["region"],
                    "city": target_weights["city"],
                }
            return {
                "country": target_weights["country"],
                "region": target_weights["region"],
                "city": target_weights["city"],
            }

        default_schedule: Dict[int, Any] = {
            0: "country",
            1000: "region",
            2000: "city",
        }

        raw_schedule = schedule if schedule is not None else default_schedule
        cleaned: Dict[int, Dict[str, Any]] = {}

        for raw_threshold, raw_value in raw_schedule.items():
            try:
                threshold = int(raw_threshold)
            except (TypeError, ValueError):
                continue

            label = "custom"
            weights: Dict[str, float] | None = None

            if isinstance(raw_value, str):
                label = raw_value.lower()
                weights = weights_for_stage(raw_value)
            elif isinstance(raw_value, dict):
                maybe_label = raw_value.get("label")
                if isinstance(maybe_label, str):
                    label = maybe_label.lower()
                raw_weights = raw_value.get("weights", raw_value)
                if isinstance(raw_weights, dict):
                    weights = {
                        "country": float(raw_weights.get("country", target_weights["country"])),
                        "region": float(raw_weights.get("region", 0.0)),
                        "city": float(raw_weights.get("city", 0.0)),
                    }

            if weights is None:
                continue

            cleaned[threshold] = {"label": label, "weights": dict(weights)}

        if not cleaned:
            cleaned[0] = {"label": "full", "weights": dict(target_weights)}

        return dict(sorted(cleaned.items()))

    def _select_active_config(self) -> Dict[str, Any]:
        """Select the active reward configuration based on episode count."""
        config = None
        for threshold, cfg in self.weight_schedule.items():
            if self.episode_count >= threshold:
                config = cfg
        if config is None:
            # Fallback to smallest threshold
            config = next(iter(self.weight_schedule.values()))
        return config

    def reset(self, idx):
        item = self.dataset[idx % len(self.dataset)]

        # Load image
        image = Image.open(item["image_path"]).convert("RGB")
        active_config = self._select_active_config()
        weights = active_config.get("weights", self.target_weights)
        stage = active_config.get("label", "full")
        self._current_stage = stage
        question = self.prompt_template

        state = OSV5MState(
            dataset_idx=idx % len(self.dataset),
            image=image,
            latitude=item["latitude"],
            longitude=item["longitude"],
            country=item["country"],
            region=item["region"],
            sub_region=item["sub_region"],
            city=item["city"],
            prediction_mode=stage,
            reward_weights=dict(weights),
            reward_stage=stage,
        )

        obs = OSV5MPrompt(
            question=question,
            image=image,
            image_path=item["image_path"],
            latitude=item["latitude"],
            longitude=item["longitude"],
            country=item["country"],
            region=item["region"],
            sub_region=item["sub_region"],
            city=item["city"],
            prediction_mode=stage,
            reward_weights=dict(weights),
            reward_stage=stage,
        )

        self.episode_count += 1

        return state, obs

    def step(self, state, action_tokens):
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        response_text = decode_tokens(self.tokenizer, action_tokens).strip()

        # Parse prediction using text extraction heuristics
        reward = 0.0
        stage = getattr(state, "reward_stage", None)
        if stage is None:
            stage = getattr(state, "prediction_mode", None)
        if stage is None:
            stage = self._current_stage
        weights = getattr(state, "reward_weights", self.target_weights)
        info = {
            "response": response_text,
            "mode": stage,
            "reward_stage": stage,
            "reward_weights": dict(weights),
        }

        # Always add ground truth to info for debugging
        info["ground_truth"] = {
            "country": state.country,
            "region": state.region,
            "city": state.city,
            "coords": (state.latitude, state.longitude)
        }

        # Extract location information from structured response formats
        parsed, format_sources = self._parse_structured_response(response_text)

        # 2) Compute hierarchical reward and optional geodesic shaping
        gt_country = state.country or ""
        gt_region = state.region or ""
        gt_city = state.city or ""
        gt_coords = (state.latitude, state.longitude)

        # Normalize predictions
        pred_country = self._normalize_country(parsed.get("country")) if parsed.get("country") is not None else None
        # Pass the predicted country to help with region/city normalization
        pred_region = self._normalize_region_or_city(parsed.get("region"), pred_country) if parsed.get("region") is not None else None
        pred_city = self._normalize_region_or_city(parsed.get("city"), pred_country) if parsed.get("city") is not None else None

        # Normalize ground truth
        gt_country_norm = (gt_country.upper() if gt_country else None)
        # Use the ground truth country for normalizing GT regions/cities
        gt_region_norm = self._normalize_region_or_city(gt_region, gt_country_norm) if gt_region else None
        gt_city_norm = self._normalize_region_or_city(gt_city, gt_country_norm) if gt_city else None

        # Hierarchical score
        hier = 0.0
        active_country_weight = float(weights.get("country", self.target_weights["country"]))
        active_region_weight = float(weights.get("region", 0.0))
        active_city_weight = float(weights.get("city", 0.0))
        if pred_country and gt_country_norm and pred_country == gt_country_norm:
            hier += max(active_country_weight, 0.0)
        if pred_region and gt_region_norm and pred_region == gt_region_norm:
            hier += max(active_region_weight, 0.0)
        if pred_city and gt_city_norm and pred_city == gt_city_norm:
            hier += max(active_city_weight, 0.0)

        # Distance shaping if lat/lon present
        r_geo: Optional[float] = None
        pred_lat = parsed.get("lat")
        pred_lon = parsed.get("lon")
        if isinstance(pred_lat, (int, float)) and isinstance(pred_lon, (int, float)):
            if -90 <= float(pred_lat) <= 90 and -180 <= float(pred_lon) <= 180:
                distance_km = self._haversine_distance(gt_coords[0], gt_coords[1], float(pred_lat), float(pred_lon))
                r_geo = exp(-float(distance_km) / max(self.geo_decay_km, 1e-6))
                info["predicted_coords"] = (float(pred_lat), float(pred_lon))
                info["distance_km"] = float(distance_km)
        info["correct_coords"] = (state.latitude, state.longitude)

        # Final reward components
        if self.use_geo_shaping and r_geo is not None:
            reward = max(hier, r_geo)
        else:
            reward = hier

        # Bonus reward for structured formatting
        parsed_fields = {k for k in parsed.keys() if k in {"city", "region", "country", "lat", "lon"}}
        format_bonus = 0.05 if format_sources and len(parsed_fields) >= 2 else 0.0
        if format_bonus > 0:
            info["format_bonus"] = float(format_bonus)
            info["format_sources"] = sorted(format_sources)
            info["structured_fields"] = sorted(parsed_fields)

        reward = float(max(min(reward + format_bonus, 1.0), 0.0))

        # Per-mode logging of predicted/correct for eval clarity
        if active_region_weight <= 0 and active_city_weight <= 0:
            info["predicted"] = pred_country
            info["correct"] = gt_country_norm
        elif active_city_weight <= 0:
            info["predicted"] = parsed.get("region")
            info["correct"] = state.region
        elif active_city_weight > 0:
            info["predicted"] = parsed.get("city")
            info["correct"] = state.city
        else:
            # already logged coords above
            pass

        # Also include full parsed dict for downstream debugging
        info["parsed"] = parsed

        return state, [], reward, True, info

    # ---------------------------- Normalizers ---------------------------- #
    def _parse_structured_response(self, text: str) -> Tuple[Dict[str, Any], set[str]]:
        """Parse structured predictions (JSON, key-value, CSV, list formats).

        Returns both the parsed fields and a set describing which structured
        patterns were detected so formatting rewards can be applied.
        """
        import json

        parsed: Dict[str, Any] = {}
        structure_hits: set[str] = set()
        if not text:
            return parsed, structure_hits

        # Extract JSON from code blocks or raw JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not json_match:
            # Try to find raw JSON without code blocks
            json_match = re.search(r'(\{.*?\})', text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                structure_hits.add("json")

                # Extract fields with flexible key names
                if "description" in data:
                    parsed["description"] = str(data["description"])
                if "city" in data:
                    parsed["city"] = str(data["city"])
                if "region" in data or "state" in data or "province" in data:
                    parsed["region"] = str(data.get("region") or data.get("state") or data.get("province"))
                if "country" in data:
                    parsed["country"] = str(data["country"])

                # Handle coordinates - could be array or separate lat/lon
                if "coordinates" in data and isinstance(data["coordinates"], list) and len(data["coordinates"]) >= 2:
                    parsed["lat"] = float(data["coordinates"][0])
                    parsed["lon"] = float(data["coordinates"][1])
                elif "latitude" in data and "longitude" in data:
                    parsed["lat"] = float(data["latitude"])
                    parsed["lon"] = float(data["longitude"])

            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        def canonical_field(key: str) -> Optional[str]:
            lowered = key.strip().lower()
            lowered = lowered.replace("-", " ")
            lowered = lowered.replace("_", " ")
            lowered = lowered.replace(".", " ")
            lowered = re.sub(r"\s+", " ", lowered)
            if "latitude" in lowered or lowered.strip() in {"lat"}:
                return "lat"
            if "longitude" in lowered or lowered.strip() in {"lon", "long", "lng"}:
                return "lon"
            if "coordinate" in lowered:
                return "coords"
            if "country" in lowered:
                return "country"
            if "city" in lowered:
                return "city"
            if "region" in lowered or "state" in lowered or "province" in lowered:
                return "region"
            return None

        def merge_value(key: str, value):
            if value is None:
                return
            if key in {"city", "region", "country"}:
                clean_val = str(value).strip().strip("`\"' \t,;")
                if clean_val and key not in parsed:
                    parsed[key] = clean_val
                return
            if key in {"lat", "lon"}:
                if key in parsed:
                    return
                match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
                if not match:
                    return
                num_str = match.group(0).replace(",", ".")
                try:
                    parsed[key] = float(num_str)
                except (TypeError, ValueError):
                    pass
                return
            if key == "coords":
                match_vals = re.findall(r"-?\d+(?:[.,]\d+)?", str(value))
                if len(match_vals) >= 2:
                    if "lat" not in parsed:
                        try:
                            parsed["lat"] = float(match_vals[0].replace(",", "."))
                        except ValueError:
                            pass
                    if "lon" not in parsed:
                        try:
                            parsed["lon"] = float(match_vals[1].replace(",", "."))
                        except ValueError:
                            pass

        def parse_key_value(candidate_text: str):
            matched = False
            for key, value in re.findall(
                r'(?im)^\s*(?:[-*+\u2022]\s*|\d+[\).\s-]+\s*)?(?:"|`)?([A-Za-z][A-Za-z0-9\s\-/_.]*?)(?:"|`)?\s*[:=\-]\s*([^\n\r]+)',
                candidate_text,
            ):
                canonical = canonical_field(key)
                if canonical:
                    merge_value(canonical, value)
                    matched = True
            if matched:
                structure_hits.add("key_value")

        def parse_csv_like(candidate_text: str):
            try:
                reader = list(csv.reader(StringIO(candidate_text)))
            except csv.Error:
                return

            header_row = None
            data_row = None
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                normalized = [cell.strip().lower() for cell in row]
                if any(
                    any(token in cell for token in ("city", "region", "state", "province", "country", "lat", "lon"))
                    for cell in normalized
                ):
                    header_row = row
                    break
            if header_row is None:
                return
            header_idx = reader.index(header_row)
            for row in reader[header_idx + 1 :]:
                if row and any(cell.strip() for cell in row):
                    data_row = row
                    break
            if data_row is None:
                return
            structure_hits.add("csv")
            for key_cell, value_cell in zip(header_row, data_row):
                canonical = canonical_field(key_cell)
                if canonical:
                    merge_value(canonical, value_cell)

        def parse_space_separated(candidate_text: str):
            matched = False
            for line in candidate_text.splitlines():
                stripped = line.strip()
                if not stripped or len(stripped) > 120:
                    continue
                sep_match = re.match(
                    r'^\s*(?:[-*+\u2022]\s*|\d+[\).\s-]+\s*)?(?:"|`)?([A-Za-z][A-Za-z0-9\s\-/_.]*?)(?:"|`)?\s+([^\s].*)$',
                    stripped,
                )
                if not sep_match:
                    continue
                key, value = sep_match.groups()
                canonical = canonical_field(key)
                if not canonical:
                    continue
                if canonical in {"city", "region", "country"} and len(value.strip()) > 80:
                    continue
                merge_value(canonical, value)
                matched = True
            if matched:
                structure_hits.add("space")

        candidate_strings = []
        candidate_strings.extend([
            block.strip()
            for block in re.findall(r"```(?:[a-zA-Z0-9]+)?\s*([\s\S]*?)```", text)
            if block.strip()
        ])
        candidate_strings.append(text.strip())

        for candidate in candidate_strings:
            if not candidate:
                continue
            parse_key_value(candidate)
            parse_space_separated(candidate)
            if "," in candidate or ";" in candidate or "\t" in candidate:
                parse_csv_like(candidate)

        return parsed, structure_hits

    def _normalize_country(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        x = s.strip().upper()
        # Map common names/aliases to ISO2 (check this first, even for 2-letter codes)
        country_map = {
            "UNITED STATES": "US", "USA": "US", "US": "US", "AMERICA": "US",
            "UNITED KINGDOM": "GB", "UK": "GB", "BRITAIN": "GB", "ENGLAND": "GB",
            "FRANCE": "FR", "GERMANY": "DE", "ITALY": "IT",
            "SPAIN": "ES", "JAPAN": "JP", "CHINA": "CN",
            "INDIA": "IN", "BRAZIL": "BR", "CANADA": "CA",
            "AUSTRALIA": "AU", "RUSSIA": "RU", "MEXICO": "MX",
            "NETHERLANDS": "NL", "HOLLAND": "NL",
            "SWEDEN": "SE", "NORWAY": "NO", "DENMARK": "DK",
            "BOLIVIA": "BO", "CHILE": "CL", "ARGENTINA": "AR",
            "PERU": "PE", "COLOMBIA": "CO", "ECUADOR": "EC",
            "PORTUGAL": "PT", "IRELAND": "IE", "POLAND": "PL",
        }
        # Check map first
        if x in country_map:
            return country_map[x]
        # If not in map and it's a valid ISO2 code, return as-is
        if re.fullmatch(r"[A-Z]{2}", x):
            return x
        # Otherwise return None
        return None

    def _normalize_region_or_city(self, s: Optional[str], country_code: Optional[str] = None) -> Optional[str]:
        if s is None:
            return None
        x = str(s)
        # Remove diacritics
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
        # Lowercase and strip punctuation at ends
        x = x.strip().lower()
        # Remove common prefixes
        x = re.sub(r"^(state of|province of|community of|autonomous community of|region of|county of|city of)\s+", "", x)
        # Collapse whitespace
        x = re.sub(r"\s+", " ", x)

        # Check country-specific aliases if country provided
        if country_code:
            aliases = self._get_region_city_aliases()
            if country_code in aliases:
                country_aliases = aliases[country_code]
                # Check if the normalized string is an alias
                if x in country_aliases:
                    x = country_aliases[x]

        return x

    def _get_region_city_aliases(self) -> Dict[str, Dict[str, str]]:
        """Get country-specific aliases for regions and cities.

        Returns a dict mapping: {country_code: {alias: canonical_name}}
        This handles common variations, abbreviations, and alternate names.
        """
        return {
            # United States - state abbreviations and common variations
            "US": {
                # State abbreviations
                "ca": "california", "cal": "california", "calif": "california",
                "ny": "new york", "n.y.": "new york", "nyc": "new york city",
                "tx": "texas", "tex": "texas",
                "fl": "florida", "fla": "florida",
                "il": "illinois", "ill": "illinois",
                "pa": "pennsylvania", "penn": "pennsylvania",
                "oh": "ohio",
                "ga": "georgia",
                "nc": "north carolina", "n carolina": "north carolina",
                "mi": "michigan", "mich": "michigan",
                "nj": "new jersey", "n jersey": "new jersey",
                "va": "virginia",
                "wa": "washington",
                "az": "arizona", "ariz": "arizona",
                "ma": "massachusetts", "mass": "massachusetts",
                "tn": "tennessee", "tenn": "tennessee",
                "in": "indiana", "ind": "indiana",
                "mo": "missouri",
                "md": "maryland",
                "wi": "wisconsin", "wis": "wisconsin", "wisc": "wisconsin",
                "co": "colorado", "colo": "colorado",
                "mn": "minnesota", "minn": "minnesota",
                "sc": "south carolina", "s carolina": "south carolina",
                "al": "alabama", "ala": "alabama",
                "la": "louisiana",
                "ky": "kentucky",
                "or": "oregon", "ore": "oregon",
                "ok": "oklahoma", "okla": "oklahoma",
                "ct": "connecticut", "conn": "connecticut",
                "ut": "utah",
                "ia": "iowa",
                "nv": "nevada", "nev": "nevada",
                "ar": "arkansas", "ark": "arkansas",
                "ms": "mississippi", "miss": "mississippi",
                "ks": "kansas", "kan": "kansas",
                "nm": "new mexico", "n mexico": "new mexico",
                "ne": "nebraska", "neb": "nebraska", "nebr": "nebraska",
                "wv": "west virginia", "w virginia": "west virginia",
                "id": "idaho", "ida": "idaho",
                "hi": "hawaii",
                "nh": "new hampshire", "n hampshire": "new hampshire",
                "me": "maine",
                "ri": "rhode island", "r island": "rhode island",
                "mt": "montana", "mont": "montana",
                "de": "delaware", "del": "delaware",
                "sd": "south dakota", "s dakota": "south dakota",
                "nd": "north dakota", "n dakota": "north dakota",
                "ak": "alaska", "alas": "alaska",
                "vt": "vermont",
                "wy": "wyoming", "wyo": "wyoming",
                # Common city aliases
                "la": "los angeles", "l.a.": "los angeles",
                "sf": "san francisco", "san fran": "san francisco", "frisco": "san francisco",
                "vegas": "las vegas",
                "nola": "new orleans",
                "philly": "philadelphia",
                "dc": "washington dc", "washington d.c.": "washington dc",
            },
            # United Kingdom - common variations
            "GB": {
                "london": "greater london",
                "manchester": "greater manchester",
                "bham": "birmingham", "b'ham": "birmingham",
                "yorks": "yorkshire",
                "lancs": "lancashire",
                "soton": "southampton",
                "wolves": "wolverhampton",
            },
            # Spain - autonomous communities and common variations
            "ES": {
                "cataluna": "catalonia", "catalunya": "catalonia",
                "balearics": "balearic islands", "baleares": "balearic islands",
                "islas baleares": "balearic islands",
                "canaries": "canary islands", "canarias": "canary islands",
                "islas canarias": "canary islands",
                "valencia": "valencian community", "comunidad valenciana": "valencian community",
                "andalusia": "andalucia",
                "bcn": "barcelona", "barna": "barcelona",
                "mad": "madrid",
                "vlc": "valencia",
                "sev": "seville", "sevilla": "seville",
                "zar": "zaragoza", "saragossa": "zaragoza",
                "palm": "palma", "palma de mallorca": "palma",
            },
            # Germany - state variations
            "DE": {
                "bavaria": "bayern", "baviera": "bayern",
                "nrw": "north rhine westphalia", "nordrhein westfalen": "north rhine westphalia",
                "bw": "baden wurttemberg", "baden-wurttemberg": "baden wurttemberg",
                "frankfurt": "frankfurt am main", "frankfurt/main": "frankfurt am main",
            },
            # France - region variations
            "FR": {
                "ile de france": "ile-de-france", "idf": "ile-de-france",
                "provence alpes cote d'azur": "provence-alpes-cote d'azur", "paca": "provence-alpes-cote d'azur",
                "rhone alpes": "auvergne-rhone-alpes",
                "nord pas de calais": "hauts-de-france",
                "marseille": "marseilles",
            },
            # Canada - province abbreviations
            "CA": {
                "on": "ontario", "ont": "ontario",
                "qc": "quebec", "que": "quebec",
                "bc": "british columbia", "b.c.": "british columbia",
                "ab": "alberta", "alta": "alberta",
                "mb": "manitoba", "man": "manitoba",
                "sk": "saskatchewan", "sask": "saskatchewan",
                "ns": "nova scotia", "n.s.": "nova scotia",
                "nb": "new brunswick", "n.b.": "new brunswick",
                "nf": "newfoundland", "nfld": "newfoundland", "nl": "newfoundland and labrador",
                "pe": "prince edward island", "pei": "prince edward island", "p.e.i.": "prince edward island",
                "nt": "northwest territories", "nwt": "northwest territories",
                "yt": "yukon", "yk": "yukon",
                "nu": "nunavut",
                # City aliases
                "mtl": "montreal",
                "van": "vancouver",
                "yyc": "calgary",
                "yyz": "toronto",
                "yvr": "vancouver",
                "yul": "montreal",
                "yow": "ottawa",
                "yeg": "edmonton",
            },
            # Australia - state abbreviations
            "AU": {
                "nsw": "new south wales", "n.s.w.": "new south wales",
                "vic": "victoria", "vict": "victoria",
                "qld": "queensland", "queensl": "queensland",
                "wa": "western australia", "w.a.": "western australia",
                "sa": "south australia", "s.a.": "south australia",
                "tas": "tasmania", "tassie": "tasmania",
                "act": "australian capital territory", "a.c.t.": "australian capital territory",
                "nt": "northern territory", "n.t.": "northern territory",
                # City aliases
                "melb": "melbourne",
                "syd": "sydney",
                "bris": "brisbane", "brissie": "brisbane",
                "adl": "adelaide",
            },
            # Brazil - state abbreviations
            "BR": {
                "sp": "sao paulo",
                "rj": "rio de janeiro", "rio": "rio de janeiro",
                "mg": "minas gerais",
                "ba": "bahia",
                "rs": "rio grande do sul",
                "pr": "parana",
                "pe": "pernambuco",
                "ce": "ceara",
                "pa": "para",
                "sc": "santa catarina",
                "go": "goias",
                "pb": "paraiba",
                "ma": "maranhao",
                "es": "espirito santo",
                "am": "amazonas",
                "rn": "rio grande do norte",
                "al": "alagoas",
                "pi": "piaui",
                "mt": "mato grosso",
                "df": "distrito federal", "brasilia": "distrito federal",
                "ms": "mato grosso do sul",
                "se": "sergipe",
                "ro": "rondonia",
                "to": "tocantins",
                "ac": "acre",
                "ap": "amapa",
                "rr": "roraima",
                # City aliases
                "sampa": "sao paulo",
                "bh": "belo horizonte",
                "ssa": "salvador",
                "bsb": "brasilia",
            }
        }
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth radius in kilometers

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance
