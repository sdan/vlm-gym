from __future__ import annotations

"""
Geospot Visual Geolocation Environment

This provides geodesic rewards for the OSV5M dataset.

"""

import json
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
from PIL import Image

import pycountry  # type: ignore
from geopy.geocoders import Nominatim  # type: ignore

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


# ============================================================================
# Types & Protocols
# ============================================================================

@dataclass(frozen=True)
class Coords:
    """Geographic coordinates with validation"""
    lat: float  # [-90, 90]
    lon: float  # [-180, 180]

    def __post_init__(self):
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon}")


@dataclass(frozen=True)
class Location:
    """Location with text labels and coordinates"""
    city: str
    region: str
    country: str  # ISO-2 code preferred
    coords: Coords


@dataclass(frozen=True)
class Sample:
    """Single dataset entry"""
    idx: int
    image_path: str
    truth: Location


@dataclass(frozen=True)
class ParsedResponse:
    """Structured model output"""
    city: str | None
    region: str | None
    country: str | None
    coords: Coords | None
    raw_text: str
    format_valid: bool


@dataclass(frozen=True)
class FieldWeights:
    """Per-field scaling factors applied to reward components."""
    country: float = 1.0
    region: float = 1.0
    city: float = 1.0
    coords: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    """Immutable reward configuration for hierarchical kernels."""
    tau_country_km: float = 2000.0
    tau_region_km: float = 500.0
    tau_city_km: float = 100.0
    tau_coord_km: float = 25.0

    coord_tolerance_km: float = 25.0
    field_weights: FieldWeights = FieldWeights()



@dataclass(frozen=True)
class CurriculumStage:
    """Curriculum progression stage"""
    name: str
    episode_threshold: int
    weight_country: float
    weight_region: float
    weight_city: float


@dataclass(frozen=True)
class RewardResult:
    """Complete reward computation result"""
    reward: float

    d_country: float | None
    d_region: float | None
    d_city: float | None
    d_coords: float | None

    k_country: float
    k_region: float
    k_city: float
    k_coords: float

    active_weight_sum: float
    within_tolerance: bool


class Geocoder(Protocol):
    """Interface for text → coords geocoding"""
    def __call__(self, query: str) -> Coords | None: ...


# ============================================================================
# Constants
# ============================================================================

PROMPT_TEMPLATE = (
    "Look at the image and guess the location.\n"
    "Respond with EXACTLY these 5 lines, no extra text:\n"
    "City: <city name>\n"
    "Region: <state or region>\n"
    "Country: <country name or ISO-2 code>\n"
    "Latitude: <number between -90 and 90>\n"
    "Longitude: <number between -180 and 180>\n"
)

KEY_ALIASES = {
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

DEFAULT_CURRICULUM = [
    CurriculumStage("country_coarse", 0, 1.0, 0.0, 0.0),      # Focus country only, broad kernels
    CurriculumStage("country", 100, 0.5, 0.0, 0.0),           # Tighten country focus
    CurriculumStage("region", 300, 0.3, 0.4, 0.0),            # Add region signal
    CurriculumStage("city", 600, 0.2, 0.3, 0.3),              # Introduce city
    CurriculumStage("full", 1000, 0.15, 0.25, 0.4),           # Balanced full task
]


# ============================================================================
# Input Processing
# ============================================================================

def parse_response(text: str) -> ParsedResponse:
    """Parse structured 5-line format or fallback formats"""
    parsed = {}
    format_sources = set()

    if not text:
        return ParsedResponse(None, None, None, None, text, False)

    # Parse key-value lines
    key_pattern = re.compile(
        r'^\s*(?:[-*+\u2022]\s*)?(?P<key>[A-Za-z][A-Za-z0-9\s\-/_.]*?)\s*:\s*(?P<value>.+)$'
    )

    for line in text.splitlines():
        match = key_pattern.match(line)
        if not match:
            continue

        key_raw = match.group("key").strip().lower()
        key_raw = key_raw.strip("*_`\"' ")
        key_raw = re.sub(r"\s+", " ", key_raw)
        canonical = KEY_ALIASES.get(key_raw)

        if canonical is None:
            continue

        value_raw = match.group("value").strip()
        value_raw = value_raw.strip("`\"' \t")
        value_raw = re.sub(r"^[*_`]+", "", value_raw)
        value_raw = re.sub(r"[*_`]+$", "", value_raw)
        value_raw = value_raw.strip()

        if canonical in {"city", "region", "country"}:
            if value_raw and canonical not in parsed:
                parsed[canonical] = value_raw
                format_sources.add("key_value")
        elif canonical in {"lat", "lon"}:
            if canonical not in parsed:
                match_num = re.search(r"-?\d+(?:[.,]\d+)?", value_raw)
                if match_num:
                    try:
                        parsed[canonical] = float(match_num.group(0).replace(",", "."))
                        format_sources.add("key_value")
                    except ValueError:
                        pass

    # Fallback: inline delimited format
    inline_parsed = _parse_inline_delimited(text)
    for key, value in inline_parsed.items():
        if key not in parsed and value is not None:
            parsed[key] = value
            format_sources.add("inline_delimited")

    # Build coords if available
    coords = None
    if "lat" in parsed and "lon" in parsed:
        try:
            coords = Coords(lat=parsed["lat"], lon=parsed["lon"])
        except ValueError:
            pass

    format_valid = bool(format_sources and len(parsed) >= 2)

    return ParsedResponse(
        city=parsed.get("city"),
        region=parsed.get("region"),
        country=parsed.get("country"),
        coords=coords,
        raw_text=text,
        format_valid=format_valid,
    )


def _parse_inline_delimited(text: str) -> dict[str, str | float]:
    """Parse comma/semicolon/pipe delimited format"""
    if not text or not any(delim in text for delim in ",;|"):
        return {}

    parts = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\s*(?:[-*+\u2022]|\d+(?:[.)]))\s*", "", stripped)
        segments = [seg.strip() for seg in re.split(r"[,|;]", stripped) if seg.strip()]
        parts.extend(segments)

    if len(parts) < 3:
        return {}

    parsed = {}
    used_indices = set()

    # Extract numeric coords from end
    numeric_indices = []
    numeric_values = {}
    for idx, part in enumerate(parts):
        part_clean = re.sub(r"^[A-Za-z][A-Za-z0-9\s\-/_.]*:\s*", "", part).strip()
        if re.fullmatch(r"-?\d+(?:[.,]\d+)?", part_clean):
            try:
                numeric_values[idx] = float(part_clean.replace(",", "."))
                numeric_indices.append(idx)
            except ValueError:
                pass
        parts[idx] = part_clean

    # Check if last two are valid coords
    if len(numeric_indices) >= 2:
        lat_idx = numeric_indices[-2]
        lon_idx = numeric_indices[-1]
        if lat_idx == len(parts) - 2 and lon_idx == len(parts) - 1:
            lat_val = numeric_values[lat_idx]
            lon_val = numeric_values[lon_idx]
            if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                parsed["lat"] = lat_val
                parsed["lon"] = lon_val
                used_indices.update({lat_idx, lon_idx})

    # Remaining parts are city, region, country
    text_parts = [parts[idx] for idx in range(len(parts)) if idx not in used_indices and parts[idx]]
    for field, value in zip(["city", "region", "country"], text_parts):
        if value:
            parsed[field] = value

    return parsed


def normalize_country(s: str | None) -> str | None:
    """Normalize to ISO-3166 alpha-2 code"""
    if not s:
        return None

    raw = s.strip()
    if not raw:
        return None

    # Already ISO-2?
    upper = raw.upper()
    if re.fullmatch(r"[A-Z]{2}", upper):
        return upper

    # Lookup via pycountry
    try:
        match = pycountry.countries.lookup(raw)  # type: ignore[attr-defined]
        if match and hasattr(match, "alpha_2"):
            return str(match.alpha_2).upper()
    except (LookupError, AttributeError):
        pass

    return None


def normalize_text(s: str | None) -> str | None:
    """Normalize city/region for comparison"""
    if not s:
        return None

    x = str(s)
    # Remove diacritics
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
    x = x.strip().lower()

    # Remove common prefixes
    x = re.sub(r"^(state of|province of|community of|autonomous community of|region of|county of|city of)\s+", "", x)

    # Expand abbreviations
    x = re.sub(r"\bst\.?\s+", "saint ", x)
    x = re.sub(r"\bsankt\s+", "saint ", x)

    # Normalize punctuation
    x = x.replace('-', ' ')
    x = re.sub(r"\s+", " ", x)

    return x if x else None


def normalize_location(loc: Location) -> Location:
    """Apply normalization to all fields"""
    return Location(
        city=normalize_text(loc.city) or "",
        region=normalize_text(loc.region) or "",
        country=normalize_country(loc.country) or loc.country.upper(),
        coords=loc.coords,
    )


# ============================================================================
# Reward Computation
# ============================================================================

def haversine_km(c1: Coords, c2: Coords) -> float:
    """Calculate distance between coordinates in km"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [c1.lat, c1.lon, c2.lat, c2.lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def coord_kernel(distance_km: float, tau_km: float) -> float:
    """Exponential decay kernel: exp(-d/τ)"""
    return math.exp(-distance_km / tau_km)


def _build_query(parts: Sequence[str | None]) -> str | None:
    """Join non-empty parts into a geocoder query."""
    clean: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            clean.append(text)
    return ", ".join(clean) if clean else None


def _country_query(raw_country: str | None) -> str | None:
    """Standardize country text for geocoding."""
    if not raw_country:
        return None
    raw_country = raw_country.strip()
    if not raw_country:
        return None

    upper = raw_country.upper()
    def _preferred_country_name(country) -> str:
        """Return a geocoder-friendly country name."""
        # Prefer common, then canonical name; official names are often verbose
        for attr in ("common_name", "name", "official_name"):
            value = getattr(country, attr, None)
            if value:
                return str(value)
        return str(country)

    if re.fullmatch(r"[A-Z]{2}", upper):
        country = pycountry.countries.get(alpha_2=upper)  # type: ignore[attr-defined]
        if country:
            return _preferred_country_name(country)
    try:
        country = pycountry.countries.lookup(raw_country)  # type: ignore[attr-defined]
        if country:
            return _preferred_country_name(country)
    except LookupError:
        pass
    return raw_country


def _geocode_required(geocoder: Geocoder, query: str) -> Coords:
    coords = geocoder(query)
    if coords is None:
        raise ValueError(f"Geocoding failed for query: {query}")
    return coords


def compute_reward(
    truth: Location,
    pred: ParsedResponse,
    config: RewardConfig,
    geocoder: Geocoder,
) -> RewardResult:
    """Pure reward computation: (truth, pred, config) → result"""

    pred_country_raw = pred.country or ""
    pred_region_raw = pred.region or ""
    pred_city_raw = pred.city or ""

    truth_country_query = _country_query(truth.country)
    # Best-effort geocoding (no raise) to keep training robust
    truth_country_coords = geocoder(truth_country_query) if truth_country_query else None

    truth_region_coords = None
    if truth.region:
        truth_region_query = _build_query([truth.region, truth_country_query])
        truth_region_coords = geocoder(truth_region_query) if truth_region_query else None

    truth_city_coords = None
    if truth.city:
        truth_city_query = _build_query([truth.city, truth.region, truth_country_query])
        truth_city_coords = geocoder(truth_city_query) if truth_city_query else None

    pred_country_query = _country_query(pred_country_raw)
    pred_country_coords = geocoder(pred_country_query) if pred_country_query else None

    pred_region_coords = None
    if pred_region_raw:
        pred_region_query = _build_query([pred_region_raw, pred_country_query])
        pred_region_coords = geocoder(pred_region_query) if pred_region_query else None

    pred_city_coords = None
    if pred_city_raw:
        pred_city_query = _build_query([pred_city_raw, pred_region_raw if pred_region_raw else None, pred_country_query])
        pred_city_coords = geocoder(pred_city_query) if pred_city_query else None

    # Distances
    d_country = haversine_km(truth_country_coords, pred_country_coords) if truth_country_coords and pred_country_coords else None
    d_region = haversine_km(truth_region_coords, pred_region_coords) if truth_region_coords and pred_region_coords else None
    d_city = haversine_km(truth_city_coords, pred_city_coords) if truth_city_coords and pred_city_coords else None
    d_coords = haversine_km(truth.coords, pred.coords) if pred.coords else None

    # Kernels
    k_country = coord_kernel(d_country, config.tau_country_km) if d_country is not None else 0.0
    k_region = coord_kernel(d_region, config.tau_region_km) if d_region is not None else 0.0
    k_city = coord_kernel(d_city, config.tau_city_km) if d_city is not None else 0.0
    k_coords = coord_kernel(d_coords, config.tau_coord_km) if d_coords is not None else 0.0

    weights = config.field_weights
    weight_map = {
        "country": weights.country if d_country is not None else 0.0,
        "region": weights.region if d_region is not None else 0.0,
        "city": weights.city if d_city is not None else 0.0,
        "coords": weights.coords if d_coords is not None else 0.0,
    }

    active_weight_sum = sum(weight_map.values())
    weighted_score = (
        weight_map["country"] * k_country
        + weight_map["region"] * k_region
        + weight_map["city"] * k_city
        + weight_map["coords"] * k_coords
    )

    reward = (weighted_score / active_weight_sum) if active_weight_sum > 0 else 0.0

    return RewardResult(
        reward=reward,
        d_country=d_country,
        d_region=d_region,
        d_city=d_city,
        d_coords=d_coords,
        k_country=k_country,
        k_region=k_region,
        k_city=k_city,
        k_coords=k_coords,
        active_weight_sum=active_weight_sum,
        within_tolerance=(d_coords <= config.coord_tolerance_km) if d_coords is not None else False,
    )


# ============================================================================
# Infrastructure
# ============================================================================

class CachedGeocoder:
    """Geocoder with persistent disk cache"""
    user_agent = "vlmrl-geospot"

    def __init__(self):
        self.cache: dict[str, Coords] = {}
        self.cache_loaded = False
        self.cache_path = Path.home() / ".cache/huggingface/datasets/osv5m/geocode_cache.json"
        self._geocoder = Nominatim(user_agent=self.user_agent)

    def __call__(self, query: str) -> Coords | None:
        self._load_cache_once()

        q = query.strip()
        if not q:
            return None

        # Check cache
        if q in self.cache:
            return self.cache[q]

        # Query online
        try:
            result = self._geocoder.geocode(q, exactly_one=True, addressdetails=False, language="en", timeout=2)
            if result is None:
                return None

            lat = float(getattr(result, "latitude", None))
            lon = float(getattr(result, "longitude", None))
            coords = Coords(lat=lat, lon=lon)

            self.cache[q] = coords
            self._save_cache()
            return coords
        except Exception:
            return None

    def _load_cache_once(self):
        if self.cache_loaded:
            return

        try:
            if self.cache_path.exists():
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) == 2:
                                try:
                                    self.cache[k] = Coords(lat=float(v[0]), lon=float(v[1]))
                                except (ValueError, TypeError):
                                    pass
        except Exception:
            pass

        self.cache_loaded = True

    def _save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({k: [v.lat, v.lon] for k, v in self.cache.items()}, f)
        except Exception:
            pass


def load_dataset(split: str = "test") -> list[Sample]:
    """Load OSV5M dataset from HuggingFace cache"""
    cache_dir = Path.home() / ".cache/huggingface/datasets/osv5m"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata
    metadata_file = cache_dir / f"{split}.csv"
    if not metadata_file.exists():
        print(f"Downloading OSV5M {split} metadata...")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=f"{split}.csv",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )

    df = pd.read_csv(metadata_file, dtype={"id": str})

    # Download images
    images_dir = cache_dir / "images" / split
    if not images_dir.exists():
        print(f"Downloading OSV5M {split} images...")
        from huggingface_hub import hf_hub_download

        images_dir.mkdir(parents=True, exist_ok=True)
        num_zips = 1 if split == "test" else 2

        for i in range(num_zips):
            zip_name = str(i).zfill(2) + ".zip"
            zip_path = hf_hub_download(
                repo_id="osv5m/osv5m",
                filename=zip_name,
                subfolder=f"images/{split}",
                repo_type="dataset",
                local_dir=str(cache_dir),
            )

            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(images_dir)

    # Build sample list
    samples = []
    checked = 0
    max_checks = min(len(df), 100000)

    print(f"Searching for available images in {images_dir}...")
    for idx, row in df.iterrows():
        checked += 1
        if checked > max_checks:
            break

        img_id = row["id"]
        if split == "test":
            img_path = images_dir / "00" / f"{img_id}.jpg"
        else:
            img_folder = img_id[:2] if len(img_id) >= 2 else "00"
            img_path = images_dir / img_folder / f"{img_id}.jpg"

        if not img_path.exists():
            continue

        try:
            coords = Coords(lat=float(row["latitude"]), lon=float(row["longitude"]))
            truth = Location(
                city=str(row["city"]) if pd.notna(row["city"]) else "",
                region=str(row["region"]) if pd.notna(row["region"]) else "",
                country=str(row["country"]) if pd.notna(row["country"]) else "",
                coords=coords,
            )
            samples.append(Sample(idx=len(samples), image_path=str(img_path), truth=truth))
        except (ValueError, KeyError):
            continue

    if not samples:
        raise ValueError(f"No valid samples found for OSV5M {split} dataset")

    print(f"Loaded {len(samples)} OSV5M samples")
    return samples


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
    """Observation for reset()"""
    question: str
    image: object  # PIL Image
    image_path: str
    sample: Sample
    stage: CurriculumStage


@dataclass(frozen=True)
class OSV5MState(BaseState):
    """Frozen state for step()"""
    sample: Sample
    image: object
    stage: CurriculumStage

    def render(self) -> str:
        return "Looking at this street view image, predict the location."


class GeospotEnv(BaseEnv):
    """Geospot environment with geodesic rewards"""

    def __init__(
        self,
        tokenizer,
        split: str = "test",
        reward_config: RewardConfig | None = None,
        curriculum: list[CurriculumStage] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token_id = _resolve_eos_id(tokenizer)
        self.tokens_per_action = 256

        self.samples = load_dataset(split)
        self.num_tasks = len(self.samples)

        self._base_reward_config = reward_config or RewardConfig()
        self.reward_config = self._base_reward_config
        self.curriculum = sorted(curriculum or DEFAULT_CURRICULUM, key=lambda s: s.episode_threshold)
        self.geocoder = CachedGeocoder()

        self.episode_count = 0

    def reset(self, idx: int) -> tuple[OSV5MState, OSV5MPrompt]:
        """Load sample and return state + prompt"""
        sample = self.samples[idx % len(self.samples)]
        image = Image.open(sample.image_path).convert("RGB")
        stage = self._current_stage()

        state = OSV5MState(sample=sample, image=image, stage=stage)
        obs = OSV5MPrompt(
            question=PROMPT_TEMPLATE,
            image=image,
            image_path=sample.image_path,
            sample=sample,
            stage=stage,
        )

        self.episode_count += 1
        return state, obs

    def _config_for_stage(self, stage: CurriculumStage) -> RewardConfig:
        """Blend curriculum weights into a stage-specific reward config."""
        base = self._base_reward_config
        coords_weight = max(stage.weight_city, 0.5 * stage.weight_region)
        field_weights = FieldWeights(
            country=stage.weight_country,
            region=stage.weight_region,
            city=stage.weight_city,
            coords=coords_weight,
        )

        return replace(base, field_weights=field_weights)

    def step(self, state: OSV5MState, action_tokens):
        """Compute reward via pure functions"""
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)

        text = decode_tokens(self.tokenizer, action_tokens).strip()
        parsed = parse_response(text)
        stage_config = self._config_for_stage(state.stage)
        result = compute_reward(
            truth=state.sample.truth,
            pred=parsed,
            config=stage_config,
            geocoder=self.geocoder,
        )

        # Build info dict
        info = {
            "response": text,
            "response_len_chars": len(text),
            "response_len_tokens": len(action_tokens),
            "stage": state.stage.name,
            "stage_weights": {
                "country": state.stage.weight_country,
                "region": state.stage.weight_region,
                "city": state.stage.weight_city,
                "coords": stage_config.field_weights.coords,
            },
            "reward_field_weights": asdict(stage_config.field_weights),
            "kernel_taus": {
                "country": stage_config.tau_country_km,
                "region": stage_config.tau_region_km,
                "city": stage_config.tau_city_km,
                "coords": stage_config.tau_coord_km,
            },
            "parsed": {
                "city": parsed.city,
                "region": parsed.region,
                "country": parsed.country,
                "coords": [parsed.coords.lat, parsed.coords.lon] if parsed.coords else None,
                "format_valid": parsed.format_valid,
            },
            "ground_truth": {
                "city": state.sample.truth.city,
                "region": state.sample.truth.region,
                "country": state.sample.truth.country,
                "coords": [state.sample.truth.coords.lat, state.sample.truth.coords.lon],
            },
            **asdict(result),
        }

        return state, [], float(result.reward), True, info

    def _current_stage(self) -> CurriculumStage:
        """Select active curriculum stage"""
        for stage in reversed(self.curriculum):
            if self.episode_count >= stage.episode_threshold:
                return stage
        return self.curriculum[0]
