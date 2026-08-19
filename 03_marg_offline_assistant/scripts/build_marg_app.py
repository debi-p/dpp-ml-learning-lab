from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path


APP_NAME = "Marg"
APP_VERSION = "1.0.1"
MODEL_ID = "dpp-gita-rag-assistant-v2"
COPY_ITEMS = [
    "ask_marg.py",
    "run_marg_desktop.py",
    "requirements.txt",
    "README.md",
    "backend",
    "models",
]


class BuildError(RuntimeError):
    pass


def build_standalone_app(project_root: Path, output_dir: Path) -> Path:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    pyinstaller = _require_pyinstaller()
    work_dir = output_dir / "pyinstaller-build"
    spec_dir = output_dir / "pyinstaller-spec"
    cache_dir = output_dir / "pyinstaller-cache"
    app_path = output_dir / f"{APP_NAME}.app"

    if app_path.exists():
        _safe_rmtree(app_path)
    if work_dir.exists():
        _safe_rmtree(work_dir)
    if spec_dir.exists():
        _safe_rmtree(spec_dir)
    if cache_dir.exists():
        _safe_rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(cache_dir)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(output_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--add-data",
        f"{project_root / 'models'}:models",
    ]
    icon_path = project_root / "assets" / "marg_icon.icns"
    if icon_path.exists():
        command.extend(["--icon", str(icon_path)])
    command.append(str(project_root / "run_marg_desktop.py"))
    subprocess.run(
        command,
        check=True,
        cwd=project_root,
        env=env,
    )

    if not app_path.exists():
        raise BuildError(f"PyInstaller finished but did not create {app_path}")

    _update_standalone_info_plist(app_path / "Contents" / "Info.plist")
    marker = app_path / "Contents" / "Resources" / "standalone-runtime.txt"
    marker.write_text("Marg includes its own Python runtime and dependencies.\n", encoding="utf-8")
    _strip_macos_metadata(app_path)
    return app_path


def build_script_app_bundle(project_root: Path, output_dir: Path) -> Path:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    app_path = output_dir / f"{APP_NAME}.app"
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    app_resources = resources / "app"

    if app_path.exists():
        _safe_rmtree(app_path)

    macos.mkdir(parents=True)
    app_resources.mkdir(parents=True)

    for item_name in COPY_ITEMS:
        source = project_root / item_name
        target = app_resources / item_name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)

    _write_info_plist(contents / "Info.plist")
    _write_launcher(macos / APP_NAME)
    _strip_macos_metadata(app_path)
    return app_path


def build_app_bundle(project_root: Path, output_dir: Path) -> Path:
    return build_standalone_app(project_root, output_dir)


def build_pkg(app_path: Path, output_dir: Path, install_scope: str = "system") -> Path:
    pkg_path = output_dir / f"{APP_NAME}.pkg"
    package_root = output_dir / "pkgroot"
    component_plist = output_dir / "components.plist"
    install_location, app_relative_path = _install_paths(install_scope)
    staged_app = package_root / app_relative_path / app_path.name
    if pkg_path.exists():
        pkg_path.unlink()
    if package_root.exists():
        _safe_rmtree(package_root)

    shutil.copytree(app_path, staged_app, ignore=shutil.ignore_patterns("._*"))
    _strip_macos_metadata(package_root)
    _write_component_plist(component_plist, app_relative_path / app_path.name)

    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"
    subprocess.run(
        [
            "pkgbuild",
            "--root",
            str(package_root),
            "--install-location",
            install_location,
            "--identifier",
            "com.dpp.marg",
            "--version",
            APP_VERSION,
            "--component-plist",
            str(component_plist),
            str(pkg_path),
        ],
        check=True,
        env=env,
    )
    return pkg_path


def _install_paths(install_scope: str) -> tuple[str, Path]:
    if install_scope == "user":
        return "/Applications", Path(".")
    if install_scope == "system":
        return "/", Path("Applications")
    raise BuildError(f"Unknown install scope: {install_scope}")


def _write_component_plist(path: Path, root_relative_bundle_path: Path) -> None:
    component = {
        "BundleHasStrictIdentifier": True,
        "BundleIsRelocatable": False,
        "BundleIsVersionChecked": False,
        "BundleOverwriteAction": "upgrade",
        "RootRelativeBundlePath": str(root_relative_bundle_path),
    }
    with path.open("wb") as handle:
        plistlib.dump([component], handle)


def _strip_macos_metadata(path: Path) -> None:
    for metadata_file in path.rglob("._*"):
        metadata_file.unlink()

    xattr = shutil.which("xattr")
    if xattr:
        subprocess.run([xattr, "-cr", str(path)], check=False)


def _safe_rmtree(path: Path) -> None:
    def on_error(function, item, _exc_info):
        os.chmod(item, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        function(item)

    shutil.rmtree(path, onerror=on_error)


def _require_pyinstaller() -> str:
    try:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise BuildError(
            "PyInstaller is required for the fully standalone Marg.app. "
            "Install it with: python3 -m pip install pyinstaller"
        ) from exc
    return "PyInstaller"


def _write_info_plist(path: Path) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Marg</string>
  <key>CFBundleDisplayName</key>
  <string>Marg</string>
  <key>CFBundleExecutable</key>
  <string>Marg</string>
  <key>CFBundleIdentifier</key>
  <string>com.dpp.marg</string>
  <key>CFBundleVersion</key>
  <string>{APP_VERSION}</string>
  <key>CFBundleShortVersionString</key>
  <string>{APP_VERSION}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>10.15</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )


def _update_standalone_info_plist(path: Path) -> None:
    with path.open("rb") as handle:
        info = plistlib.load(handle)
    info["CFBundleIdentifier"] = "com.dpp.marg"
    info["CFBundleVersion"] = APP_VERSION
    info["CFBundleShortVersionString"] = APP_VERSION
    info["CFBundleName"] = APP_NAME
    info["CFBundleDisplayName"] = APP_NAME
    with path.open("wb") as handle:
        plistlib.dump(info, handle)


def _write_launcher(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
APP_DIR="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
cd "$APP_DIR"
exec /usr/bin/env python3 "$APP_DIR/run_marg_desktop.py"
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline standalone Marg macOS app and installer package.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    parser.add_argument("--pkg", action="store_true", help="Also build Marg.pkg using macOS pkgbuild.")
    parser.add_argument(
        "--install-scope",
        choices=["user", "system"],
        default="user",
        help="Install Marg.app to ~/Applications for this user, or /Applications for system-wide install.",
    )
    parser.add_argument(
        "--script-bundle",
        action="store_true",
        help="Build the old lightweight app bundle that uses the Mac's python3 runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.script_bundle:
        app_path = build_script_app_bundle(args.project_root, args.output_dir)
    else:
        app_path = build_standalone_app(args.project_root, args.output_dir)
    print(f"Built app: {app_path}")

    if args.pkg:
        pkg_path = build_pkg(app_path, args.output_dir, install_scope=args.install_scope)
        print(f"Built package: {pkg_path}")


if __name__ == "__main__":
    main()
