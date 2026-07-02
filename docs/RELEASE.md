# Releasing Fuselk Desktop

GitHub Actions packages the PySide6 control room (`fuselk gui`) for macOS, Linux, and Windows when you push a version tag. Asset filenames match the [Deepiri landing site](https://github.com/Team-Deepiri/deepiri-landing).

## Cut a release

1. Merge changes to `main`.
2. Bump `version` in `pyproject.toml` if needed.
3. Tag and push (current package version is `0.5.0`):

   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

4. Watch [Release workflow](https://github.com/Team-Deepiri/deepiri-fuselk/actions/workflows/release.yml).

## Test CI without tagging

**Actions → Release → Run workflow** on a branch. Build jobs run; publish is skipped unless the ref is a `v*` tag.

## Local packaging (optional)

```bash
poetry install --with desktop
pip install pyinstaller
bash scripts/ci/build-release.sh
ls release/
```

## Release assets

| Platform | Filename |
|----------|----------|
| macOS | `Fuselk-latest.dmg` |
| Linux | `Fuselk-latest.AppImage` |
| Windows | `Fuselk-latest-setup.exe` |

## Verify download URLs

```bash
BASE=https://github.com/Team-Deepiri/deepiri-fuselk/releases/latest/download

curl -I "$BASE/Fuselk-latest.dmg"
curl -I "$BASE/Fuselk-latest.AppImage"
curl -I "$BASE/Fuselk-latest-setup.exe"
```

## Notes

- CI sets `SKIP_FETCH=1` so dataset downloads are not run during packaging.
- Builds are **unsigned** for v1 (Gatekeeper / SmartScreen warnings expected).
- The frozen app bundles `experiments/registry.yaml` and `viz/static/` assets required by the GUI.
- After installing, smoke-test: launch app → wait for Dash/API backends → open embedded panels.
