from pathlib import Path

from mando_tools import workspace_paths


def _create_package_root(path: Path) -> None:
    (path / 'mando_tools').mkdir(parents=True)
    (path / 'launch').mkdir()
    (path / 'data').mkdir()
    (path / 'package.xml').write_text('<package/>', encoding='utf-8')


def test_workspace_root_finds_standalone_checkout(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / 'traffic_light'
    _create_package_root(package_root)
    monkeypatch.chdir(package_root / 'mando_tools')
    monkeypatch.delenv('MANDO_WS', raising=False)
    monkeypatch.delenv('MANDO_WORKSPACE', raising=False)

    assert workspace_paths.workspace_root() == package_root


def test_workspace_root_finds_ssc_nested_checkout(tmp_path, monkeypatch) -> None:
    ssc_root = tmp_path / 'SSC'
    package_root = ssc_root / 'src' / 'perception' / 'traffic_light'
    _create_package_root(package_root)
    installed_module = (
        ssc_root / 'install' / 'mando_tools' / 'lib' / 'python3.10'
        / 'site-packages' / 'mando_tools' / 'workspace_paths.py'
    )
    installed_module.parent.mkdir(parents=True)
    installed_module.touch()
    monkeypatch.chdir(installed_module.parent)
    monkeypatch.delenv('MANDO_WS', raising=False)
    monkeypatch.delenv('MANDO_WORKSPACE', raising=False)

    assert workspace_paths.workspace_root() == package_root


def test_ssc_root_environment_override_is_supported(tmp_path, monkeypatch) -> None:
    ssc_root = tmp_path / 'SSC'
    package_root = ssc_root / 'src' / 'perception' / 'traffic_light'
    _create_package_root(package_root)
    monkeypatch.setenv('MANDO_WS', str(ssc_root))

    assert workspace_paths.workspace_root() == package_root
