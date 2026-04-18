from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._command_running = False
    cli_obj._tool_start_time = 0.0
    return cli_obj


def test_periodic_tui_refresh_disabled_while_idle():
    cli_obj = _make_cli()

    assert cli_obj._periodic_tui_refresh_interval() is None


def test_periodic_tui_refresh_kept_for_command_spinner():
    cli_obj = _make_cli()
    cli_obj._command_running = True

    assert cli_obj._periodic_tui_refresh_interval() == 0.1


def test_command_running_takes_precedence_over_tool_timer():
    cli_obj = _make_cli()
    cli_obj._command_running = True
    cli_obj._tool_start_time = 1.0

    assert cli_obj._periodic_tui_refresh_interval() == 0.1


def test_periodic_tui_refresh_kept_for_tool_elapsed_timer():
    cli_obj = _make_cli()
    cli_obj._tool_start_time = 1.0

    assert cli_obj._periodic_tui_refresh_interval() == 0.15
