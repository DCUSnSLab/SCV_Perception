from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_BAG_PROFILE = 'stop_points'
_BAG_PROFILES = {
    'mando_ros2': {
        'bag_dir': 'mando_ros2',
        'image_topic': '/zed_node/left/image_rect_color',
    },
    'stop_points': {
        'bag_dir': 'stop_points',
        'image_topic': '/zed/zed_node/left/image_rect_color',
    },
}
_FALLBACK_IMAGE_TOPIC = _BAG_PROFILES['mando_ros2']['image_topic']
_DEFAULT_RUNTIME_IMAGE_TOPIC = '/mando/input/image'
# In SSC this ROS package is kept as a nested perception component.  Installed
# Python/launch files live under install/mando_tools, so walking parents alone
# cannot see the package's model, data, and optional local dependencies.
_SSC_TRAFFIC_LIGHT_RELATIVE_PATH = Path('src/perception/traffic_light')


def _iter_search_roots(start: Path) -> list[Path]:
    resolved = start.expanduser().resolve()
    anchor = resolved.parent if resolved.is_file() else resolved
    return [anchor, *anchor.parents]


def _is_workspace_root(path: Path) -> bool:
    # Model weights are optional runtime assets and are intentionally not kept
    # in Git, so their absence must not prevent bag-only tools from starting.
    return (
        (path / 'mando_tools').is_dir()
        and (path / 'launch').is_dir()
        and (path / 'data').is_dir()
        and (path / 'package.xml').is_file()
    )


def _iter_workspace_candidates(start: Path) -> list[Path]:
    """Return standalone and SSC-nested package roots reachable from start."""
    candidates = []
    for root in _iter_search_roots(start):
        candidates.append(root)
        candidates.append(root / _SSC_TRAFFIC_LIGHT_RELATIVE_PATH)
    return candidates


def workspace_root() -> Path:
    env_candidates = [
        os.environ.get('MANDO_WS'),
        os.environ.get('MANDO_WORKSPACE'),
    ]
    search_starts = [Path.cwd(), Path(__file__)]

    for candidate in env_candidates:
        if candidate:
            search_starts.insert(0, Path(candidate))

    checked = set()
    for start in search_starts:
        for root in _iter_workspace_candidates(start):
            if root in checked:
                continue
            checked.add(root)
            if _is_workspace_root(root):
                return root

    raise RuntimeError('Could not locate the mando workspace root.')


def _default_bag_selection() -> str:
    return os.environ.get('MANDO_BAG', _DEFAULT_BAG_PROFILE)


def _bag_profile_path(root: Path, bag_name: str) -> Path:
    profile = _BAG_PROFILES[bag_name]
    bag_dir = str(profile['bag_dir'])
    return root / 'data' / 'bags' / bag_dir


def resolve_bag_path(selection: str | os.PathLike[str] | None = None) -> Path:
    root = workspace_root()
    raw = os.fspath(selection) if selection is not None else _default_bag_selection()

    if raw in _BAG_PROFILES:
        return _bag_profile_path(root, raw)

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (root / candidate).resolve()


def default_bag_path() -> Path:
    return resolve_bag_path()


def default_image_topic(selection: str | os.PathLike[str] | None = None) -> str:
    raw = os.fspath(selection) if selection is not None else _default_bag_selection()
    if raw in _BAG_PROFILES:
        return str(_BAG_PROFILES[raw]['image_topic'])

    resolved = resolve_bag_path(raw)
    root = workspace_root()
    for bag_name, profile in _BAG_PROFILES.items():
        if resolved == _bag_profile_path(root, bag_name).resolve():
            return str(profile['image_topic'])
    return str(_FALLBACK_IMAGE_TOPIC)


def default_runtime_image_topic() -> str:
    return os.environ.get('MANDO_IMAGE_TOPIC', _DEFAULT_RUNTIME_IMAGE_TOPIC).strip() or (
        _DEFAULT_RUNTIME_IMAGE_TOPIC
    )


def default_model_path() -> Path:
    return workspace_root() / 'model' / 'best.pt'


def default_inference_device() -> str:
    override = os.environ.get('MANDO_DEVICE', '').strip()
    if override:
        return resolve_inference_device(override)

    return 'cuda:0' if _cuda_available() else 'cpu'


def resolve_inference_device(selection: str | os.PathLike[str] | None = None) -> str:
    raw = os.fspath(selection).strip() if selection is not None else ''
    if not raw or raw.lower() == 'auto':
        return default_inference_device()
    if raw.lower().startswith('cuda') and not _cuda_available():
        return 'cpu'
    return raw


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def local_python_deps_path() -> Path:
    root_deps = workspace_root() / '.deps'
    if root_deps.exists():
        return root_deps
    return workspace_root() / 'mando_tools' / '.deps'
