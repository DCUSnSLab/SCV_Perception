from pathlib import Path

import pytest

from mando_tools import workspace_paths


def make_package(root):
    (root / 'mando_tools').mkdir(parents=True)
    (root / 'launch').mkdir()
    (root / 'package.xml').touch()
    return root


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv('MANDO_WS', raising=False)
    monkeypatch.delenv('MANDO_WORKSPACE', raising=False)
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    return tmp_path / 'workspace'


@pytest.mark.parametrize('install_path, source_path', [
    ('install/mando_tools/lib/python3.10/site-packages/mando_tools', 'src/perception/traffic_light'),
    ('install/lib/python3.10/site-packages/mando_tools', 'src/perception/traffic_light'),
    ('src/perception/install/mando_tools/lib/python3.10/site-packages/mando_tools', 'src/perception/traffic_light'),
    ('install/mando_tools/lib/python3.10/site-packages/mando_tools', 'src/traffic_light'),
    ('install/mando_tools/lib/python3.10/site-packages/mando_tools', 'traffic_light'),
])
def test_installed_module_finds_source_without_assets(isolated_workspace, monkeypatch, install_path, source_path):
    package = make_package(isolated_workspace / source_path)
    module = isolated_workspace / install_path / 'workspace_paths.py'
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.setattr(workspace_paths, '__file__', str(module))
    assert workspace_paths.workspace_root_or_none() == package
    assert workspace_paths.default_model_path() == package / 'model/best.pt'


def test_source_module_precedes_unrelated_current_package(isolated_workspace, monkeypatch):
    package = make_package(isolated_workspace / 'src/perception/traffic_light')
    other = make_package(isolated_workspace / 'other')
    module = package / 'mando_tools/workspace_paths.py'
    module.touch()
    monkeypatch.setattr(workspace_paths, '__file__', str(module))
    monkeypatch.chdir(other)
    assert workspace_paths.workspace_root_or_none() == package


def test_explicit_workspace_precedes_module_and_legacy_env(isolated_workspace, monkeypatch):
    package = make_package(isolated_workspace / 'src/perception/traffic_light')
    preferred = make_package(isolated_workspace / 'preferred')
    monkeypatch.setattr(workspace_paths, '__file__', str(package / 'mando_tools/workspace_paths.py'))
    monkeypatch.setenv('MANDO_WS', str(preferred))
    monkeypatch.setenv('MANDO_WORKSPACE', str(package))
    assert workspace_paths.workspace_root_or_none() == preferred


def test_workspace_env_accepts_colcon_root(isolated_workspace, monkeypatch):
    package = make_package(isolated_workspace / 'src/perception/traffic_light')
    monkeypatch.setenv('MANDO_WS', str(isolated_workspace))
    assert workspace_paths.workspace_root_or_none() == package


def test_missing_source_returns_none(isolated_workspace, monkeypatch):
    monkeypatch.setattr(workspace_paths, '__file__', str(isolated_workspace / 'install/mando_tools/workspace_paths.py'))
    assert workspace_paths.workspace_root_or_none() is None
    with pytest.raises(RuntimeError, match='MANDO_WS'):
        workspace_paths.workspace_root()
