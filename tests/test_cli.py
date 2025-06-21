import types
import os
import sys
from click.testing import CliRunner

sys.path.insert(0, os.getcwd())

# Stub keyring before importing cli
keyring_stub = types.SimpleNamespace(
    set_password=lambda *a, **k: None,
    delete_password=lambda *a, **k: None,
    errors=types.SimpleNamespace(PasswordDeleteError=Exception),
)
sys.modules['keyring'] = keyring_stub

# Stub rich modules used in cli.setup
rich_prompt_stub = types.SimpleNamespace(Prompt=types.SimpleNamespace(ask=lambda *a, **k: 'user'),
                                         IntPrompt=types.SimpleNamespace(ask=lambda *a, **k: '1'))
rich_console_stub = types.SimpleNamespace(Console=lambda: types.SimpleNamespace(print=lambda *a, **k: None))
sys.modules.setdefault('rich', types.ModuleType('rich'))
sys.modules['rich.prompt'] = rich_prompt_stub
sys.modules['rich.console'] = rich_console_stub
sys.modules['rich.table'] = types.SimpleNamespace(Table=lambda *a, **k: object())

# Create a stub tastytrade module to satisfy imports in cli.setup
tastytrade_stub = types.ModuleType('tastytrade')
tastytrade_stub.Session = lambda *a, **k: object()
tastytrade_stub.Account = types.SimpleNamespace(get=lambda session: [types.SimpleNamespace(account_number='A1')])
sys.modules['tastytrade'] = tastytrade_stub

import tasty_agent.cli as cli


def test_main_runs_server(monkeypatch):
    runner = CliRunner()
    called = []
    mcp_stub = types.SimpleNamespace(run=lambda: called.append(True))
    monkeypatch.setitem(sys.modules, 'tasty_agent.server', types.SimpleNamespace(mcp=mcp_stub))
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert called


def test_setup_single_account(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(cli, 'getpass', lambda prompt: 'pass')

    set_calls = []

    def fake_set(service, key, value):
        set_calls.append((service, key, value))

    monkeypatch.setattr(cli.keyring, 'set_password', fake_set)



    result = runner.invoke(cli.setup)
    assert result.exit_code == 0
    assert ('tastytrade', 'username', 'user') in set_calls
    assert ('tastytrade', 'password', 'pass') in set_calls
    assert ('tastytrade', 'account_id', 'A1') in set_calls
