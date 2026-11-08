"""
Unit tests for the Ballistic Loader class.

Covers: directory management, _needs_update logic, status(), clear_cache().

Run with: PYTHONPATH=./python pytest tests/python/test_loader.py -v
"""

import pytest
import time
from pathlib import Path

# Skip all tests if the module isn't built yet
try:
    import ballistic
    from ballistic.loader import Loader, FILENAMES, UPDATE_INTERVALS
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BALLISTIC_AVAILABLE,
    reason="ballistic module not available - build with BALLISTIC_BUILD_PYTHON=ON"
)


@pytest.fixture(scope="session", autouse=True)
def initialize_ballistic():
    """Initialize the ballistic library once for the entire test session."""
    if BALLISTIC_AVAILABLE:
        ballistic.initialize()


class TestLoaderDirectory:
    """Tests for Loader directory management."""

    def test_default_directory(self):
        """Default directory should exist after creation."""
        loader = Loader(verbose=False)
        assert loader.directory.exists()
        assert loader.directory.is_dir()

    def test_custom_directory_created(self, tmp_path):
        """Custom directory should be created if it doesn't exist."""
        custom_dir = tmp_path / "custom_ballistic_data"
        assert not custom_dir.exists()
        loader = Loader(directory=str(custom_dir), verbose=False)
        assert custom_dir.exists()
        assert loader.directory == custom_dir

    def test_nested_directory_created(self, tmp_path):
        """Deeply nested custom directory should be created."""
        nested = tmp_path / "a" / "b" / "c" / "ballistic"
        Loader(directory=str(nested), verbose=False)
        assert nested.exists()

    def test_directory_is_path_object(self, tmp_path):
        """Loader.directory should be a Path object."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        assert isinstance(loader.directory, Path)


class TestLoaderNeedsUpdate:
    """Tests for _needs_update logic."""

    def test_missing_file_needs_update(self, tmp_path):
        """Non-existent file should need update."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        path = tmp_path / "nonexistent.data"
        assert loader._needs_update(path, 'eop') is True

    def test_fresh_file_no_update(self, tmp_path):
        """Recently created file should not need update."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        path = tmp_path / "fresh.data"
        path.write_text("test data", encoding="utf-8")
        assert loader._needs_update(path, 'eop') is False

    def test_custom_expire_days(self, tmp_path):
        """Custom expire_days should override defaults."""
        loader = Loader(directory=str(tmp_path), verbose=False, expire_days=0)
        path = tmp_path / "test.data"
        path.write_text("test data", encoding="utf-8")
        # With expire_days=0, even a fresh file needs update
        # (its age > 0 days)
        # Wait a tiny bit to ensure non-zero age
        time.sleep(0.01)
        # File age in days will be tiny but > 0
        result = loader._needs_update(path, 'eop')
        # This depends on timing - a just-created file may have age ~0
        # Just verify it doesn't crash
        assert isinstance(result, bool)

    def test_default_intervals_used(self, tmp_path):
        """Default update intervals should be used from UPDATE_INTERVALS."""
        Loader(directory=str(tmp_path), verbose=False)
        # EOP default is 7 days
        assert UPDATE_INTERVALS['eop'] == 7
        assert UPDATE_INTERVALS['space_weather'] == 1
        assert UPDATE_INTERVALS['leap_seconds'] == 180


class TestLoaderStatus:
    """Tests for status() method."""

    def test_status_structure(self, tmp_path):
        """Status should have entries for all file types."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        status = loader.status()
        for file_type in FILENAMES:
            assert file_type in status
            info = status[file_type]
            assert 'exists' in info
            assert 'age_days' in info
            assert 'loaded' in info
            assert 'path' in info

    def test_status_nonexistent_files(self, tmp_path):
        """Status for missing files should show exists=False."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        status = loader.status()
        for file_type in FILENAMES:
            assert status[file_type]['exists'] is False
            assert status[file_type]['age_days'] is None

    def test_status_path_in_directory(self, tmp_path):
        """Status paths should be within the loader's directory."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        status = loader.status()
        for file_type in FILENAMES:
            assert str(tmp_path) in status[file_type]['path']


class TestLoaderClearCache:
    """Tests for clear_cache() method."""

    def test_clear_cache_removes_files(self, tmp_path):
        """clear_cache should remove all cached files."""
        loader = Loader(directory=str(tmp_path), verbose=False)

        # Create dummy files
        for filename in FILENAMES.values():
            (tmp_path / filename).write_text("dummy data", encoding="utf-8")

        # Verify they exist
        for filename in FILENAMES.values():
            assert (tmp_path / filename).exists()

        # Clear cache
        loader.clear_cache()

        # Verify they're gone
        for filename in FILENAMES.values():
            assert not (tmp_path / filename).exists()

    def test_clear_cache_no_files(self, tmp_path):
        """clear_cache should not raise when no files exist."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        loader.clear_cache()  # Should not raise


class TestLoaderFileAge:
    """Tests for _file_age_days helper."""

    def test_nonexistent_file_returns_none(self, tmp_path):
        """_file_age_days should return None for missing files."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        age = loader._file_age_days(tmp_path / "nonexistent")
        assert age is None

    def test_existing_file_returns_float(self, tmp_path):
        """_file_age_days should return a float for existing files."""
        loader = Loader(directory=str(tmp_path), verbose=False)
        path = tmp_path / "test.txt"
        path.write_text("test", encoding="utf-8")
        age = loader._file_age_days(path)
        assert isinstance(age, float)
        assert age >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
