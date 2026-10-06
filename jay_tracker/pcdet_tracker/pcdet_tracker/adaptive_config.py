"""YAML configuration helpers for the adaptive tracker prototype."""

from pathlib import Path

import yaml

from .adaptive_track_policy import AdaptivePolicyConfig
from .observation_analyzer import ObservationConfig


def load_adaptive_config(path):
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    return data.get('adaptive_tracking', data)


def _known_fields(cls, values):
    names = cls.__dataclass_fields__.keys()
    return {key: value for key, value in values.items() if key in names}


def observation_config(values):
    observation = values.get('observation', {})
    lidar = observation.get('lidar', {})
    camera = observation.get('camera', {})
    depth = observation.get('depth', {})
    merged = {}
    merged.update({f'lidar_{key}': value for key, value in lidar.items()})
    merged.update({f'camera_{key}': value for key, value in camera.items()})
    merged.update({f'depth_{key}': value for key, value in depth.items()})
    if 'class_mapping' in values:
        merged['class_mapping'] = values['class_mapping']
    return ObservationConfig(**_known_fields(ObservationConfig, merged))


def policy_config(values):
    policy = dict(values.get('policy', {}))
    policy['enable_adaptive_association'] = bool(
        values.get('enable_adaptive_association', False))
    policy['enable_adaptive_lifecycle'] = bool(
        values.get('enable_adaptive_lifecycle', False))
    return AdaptivePolicyConfig(**_known_fields(AdaptivePolicyConfig, policy))
