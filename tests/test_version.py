import os
import sys

sys.path.insert(0, os.getcwd())

from tasty_agent import _version


def test_version_string():
    assert isinstance(_version.__version__, str)
    assert _version.__version__ == _version.version
