import os
import sys

sys.path.insert(0, os.getcwd())

from tasty_agent import utils


def test_is_test_env_true(monkeypatch):
    monkeypatch.setenv('TASTYTRADE_IS_TEST', 'TRUE')
    assert utils.is_test_env() is True


def test_is_test_env_false(monkeypatch):
    monkeypatch.delenv('TASTYTRADE_IS_TEST', raising=False)
    assert utils.is_test_env() is False
