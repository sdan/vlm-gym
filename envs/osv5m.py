from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any
from pathlib import Path

import pandas as pd
from PIL import Image
import re
import unicodedata
from math import exp

try:
    import pycountry  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency required
    raise ImportError(
        "The OSV5M environment requires `pycountry`. Install it via `pip install pycountry`."
    ) from exc

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
            # Numeric lengths for aggregation in trainer
            "response_len_chars": int(len(response_text)),
            "response_len_tokens": int(len(action_tokens)),
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
        pred_region = self._normalize_region_or_city(parsed.get("region")) if parsed.get("region") is not None else None
        pred_city = self._normalize_region_or_city(parsed.get("city")) if parsed.get("city") is not None else None

        # Normalize ground truth
        gt_country_norm = (gt_country.upper() if gt_country else None)
        gt_region_norm = self._normalize_region_or_city(gt_region) if gt_region else None
        gt_city_norm = self._normalize_region_or_city(gt_city) if gt_city else None

        # Hierarchical score + per-target correctness flags
        hier = 0.0
        active_country_weight = float(weights.get("country", self.target_weights["country"]))
        active_region_weight = float(weights.get("region", 0.0))
        active_city_weight = float(weights.get("city", 0.0))
        country_correct = int(1 if (pred_country and gt_country_norm and pred_country == gt_country_norm) else 0)
        region_correct = int(1 if (pred_region and gt_region_norm and pred_region == gt_region_norm) else 0)
        city_correct = int(1 if (pred_city and gt_city_norm and pred_city == gt_city_norm) else 0)
        if country_correct:
            hier += max(active_country_weight, 0.0)
        if region_correct:
            hier += max(active_region_weight, 0.0)
        if city_correct:
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
                info["coords_in_range"] = int(1)
            else:
                info["coords_in_range"] = int(0)
        info["correct_coords"] = (state.latitude, state.longitude)
        # Coordinate presence/tolerance metrics
        has_coords = int(isinstance(pred_lat, (int, float)) and isinstance(pred_lon, (int, float)))
        info["has_coords"] = has_coords
        if has_coords and "distance_km" in info:
            info["coords_within_tolerance"] = int(float(info["distance_km"]) <= float(self.coord_tolerance_km))
        else:
            info["coords_within_tolerance"] = int(0)
        info["coord_tolerance_km"] = float(self.coord_tolerance_km)

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
        # Numeric flags for formatting & parsing
        info["format_ok"] = int(1 if (format_sources and len(parsed_fields) >= 2) else 0)
        info["parsed_field_count"] = int(len(parsed_fields))
        info["has_country"] = int(1 if (parsed.get("country") is not None) else 0)
        info["has_region"] = int(1 if (parsed.get("region") is not None) else 0)
        info["has_city"] = int(1 if (parsed.get("city") is not None) else 0)
        # Country value validity after normalization (did it map to ISO-2?)
        info["country_valid"] = int(1 if (parsed.get("country") is not None and pred_country is not None) else 0)

        reward = float(max(min(reward + format_bonus, 1.0), 0.0))
        # Reward component & stage logging (numeric for aggregation)
        info["hier_reward"] = float(hier)
        info["geo_reward"] = float(r_geo) if r_geo is not None else float(0.0)
        info["final_reward"] = float(reward)
        stage_label = str(stage).lower() if isinstance(stage, str) else str(stage)
        stage_to_idx = {"country": 0, "region": 1, "city": 2, "coords": 3, "full": 9, "custom": 8}
        info["stage_idx"] = int(stage_to_idx.get(stage_label, 9))
        info["stage_country_weight"] = float(active_country_weight)
        info["stage_region_weight"] = float(active_region_weight)
        info["stage_city_weight"] = float(active_city_weight)
        info["country_correct"] = int(country_correct)
        info["region_correct"] = int(region_correct)
        info["city_correct"] = int(city_correct)
        info["any_correct"] = int(1 if (country_correct or region_correct or city_correct) else 0)
        if active_region_weight <= 0 and active_city_weight <= 0:
            stage_acc = int(country_correct)
        elif active_city_weight <= 0:
            stage_acc = int(region_correct)
        elif active_city_weight > 0:
            stage_acc = int(city_correct)
        else:
            stage_acc = int(info.get("coords_within_tolerance", 0))
        info["stage_acc"] = int(stage_acc)
        # Backward/standard alias for overall correctness under current stage
        info["is_correct"] = int(stage_acc)

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
        """Parse colon-delimited key/value predictions.

        Returns both the parsed fields and a set describing which structured
        patterns were detected so formatting rewards can be applied.
        """
        parsed: Dict[str, Any] = {}
        structure_hits: set[str] = set()
        if not text:
            return parsed, structure_hits

        key_pattern = re.compile(
            r'^\s*(?:[-*+\u2022]\s*)?(?P<key>[A-Za-z][A-Za-z0-9\s\-/_.]*?)\s*:\s*(?P<value>.+)$'
        )
        key_aliases = {
            "city": "city",
            "country": "country",
            "region": "region",
            "state": "region",
            "province": "region",
            "latitude": "lat",
            "lat": "lat",
            "longitude": "lon",
            "lon": "lon",
        }

        for raw_line in text.splitlines():
            match = key_pattern.match(raw_line)
            if not match:
                continue
            raw_key = match.group("key")
            raw_value = match.group("value")

            key_clean = raw_key.strip().lower()
            key_clean = key_clean.strip("*_`\"' ")
            key_clean = re.sub(r"\s+", " ", key_clean)
            canonical = key_aliases.get(key_clean)
            if canonical is None:
                continue

            value_clean = raw_value.strip()
            value_clean = value_clean.strip("`\"' \t")
            value_clean = re.sub(r"^[*_`]+", "", value_clean)
            value_clean = re.sub(r"[*_`]+$", "", value_clean)
            value_clean = value_clean.strip()

            if canonical in {"city", "region", "country"}:
                if value_clean and canonical not in parsed:
                    parsed[canonical] = value_clean
                    structure_hits.add("key_value")
            elif canonical in {"lat", "lon"}:
                if canonical in parsed:
                    continue
                match_num = re.search(r"-?\d+(?:[.,]\d+)?", value_clean)
                if not match_num:
                    continue
                number = match_num.group(0).replace(",", ".")
                try:
                    parsed[canonical] = float(number)
                    structure_hits.add("key_value")
                except ValueError:
                    continue

        return parsed, structure_hits

    def _normalize_country(self, s: Optional[str]) -> Optional[str]:
        """Normalize country name to ISO-3166 alpha-2 code using pycountry.

        Returns the 2-letter ISO code (e.g., "US", "GB", "FR") or None if not found.
        """
        if s is None:
            return None
        raw = str(s).strip()
        if not raw:
            return None

        # If already ISO-2 format, return as-is
        upper = raw.upper()
        if re.fullmatch(r"[A-Z]{2}", upper):
            return upper

        # Use pycountry to look up country
        try:
            match = pycountry.countries.lookup(raw)  # type: ignore[attr-defined]
        except (LookupError, AttributeError):
            return None

        if match is not None and hasattr(match, "alpha_2"):
            return str(match.alpha_2).upper()

        return None

    def _normalize_region_or_city(self, s: Optional[str]) -> Optional[str]:
        """Normalize region/city name to canonical form for comparison.

        Handles diacritics, case, whitespace, and common alias canonicalization
        (e.g., "St. Petersburg" -> "saint petersburg").
        """
        if s is None:
            return None
        x = str(s)
        # Remove diacritics (e.g., "São Paulo" → "Sao Paulo")
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
        # Lowercase and strip
        x = x.strip().lower()
        # Remove common prefixes
        x = re.sub(r"^(state of|province of|community of|autonomous community of|region of|county of|city of)\s+", "", x)
        # Expand common abbreviations
        x = re.sub(r"\bst\.?\s+", "saint ", x)
        x = re.sub(r"\bsankt\s+", "saint ", x)
        # Normalize punctuation variants
        x = x.replace('-', ' ')
        # Collapse whitespace
        x = re.sub(r"\s+", " ", x)
        return x

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
