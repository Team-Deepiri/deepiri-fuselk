"""Tests for Deepiri brand assets."""

from deepiri_fuselk.viz.branding import FAVICON_SVG_PATH, LOGO_PATH, LOGO_SQUARED_PATH, branding_dir


def test_brand_assets_exist():
    assert branding_dir().is_dir()
    assert LOGO_PATH.is_file()
    assert LOGO_SQUARED_PATH.is_file()
    assert FAVICON_SVG_PATH.is_file()
    assert LOGO_PATH.stat().st_size > 10_000
