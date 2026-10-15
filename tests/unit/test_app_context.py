import signal
from unittest.mock import patch

import pytest
import requests

import manhattan_reasoning_gym as mrg
from manhattan_reasoning_gym import _client


def test_context_manager_normal_exit_releases_session():
    app = mrg.App("test_app", design="d.py", api_key="k", fpga_id=2)
    app._programmed = True

    with patch.object(
        _client, "release_session", return_value="job-reset-1"
    ) as mock_rel:
        with app:
            pass
        mock_rel.assert_called_once_with(2, "k", app.api_url)
    assert not app._programmed


def test_context_manager_keyboard_interrupt_releases_session():
    app = mrg.App("test_app", design="d.py", api_key="k", fpga_id=3)
    app._programmed = True

    with patch.object(
        _client, "release_session", return_value="job-reset-2"
    ) as mock_rel:
        with pytest.raises(KeyboardInterrupt):
            with app:
                raise KeyboardInterrupt()
        mock_rel.assert_called_once_with(3, "k", app.api_url)
    assert not app._programmed


def test_context_manager_unprogrammed_exit_does_not_call_release():
    app = mrg.App("test_app", design="d.py", api_key="k")
    assert app.fpga_id is None
    assert not app._programmed

    with patch.object(_client, "release_session") as mock_rel:
        with app:
            pass
        mock_rel.assert_not_called()


def test_program_abort_cancels_in_flight_build():
    app = mrg.App("test_app", design="d.py", api_key="k")

    with (
        patch.object(_client, "submit", return_value="build-job-123"),
        patch.object(_client, "poll_job", side_effect=KeyboardInterrupt),
        patch.object(_client, "cancel_job") as mock_cancel,
    ):
        with pytest.raises(KeyboardInterrupt):
            app._program()

        mock_cancel.assert_called_once_with("build-job-123", "k", app.api_url)
    assert not app._programmed
    assert app.fpga_id is None


def test_release_falls_back_to_reset_on_404():
    app = mrg.App("test_app", design="d.py", api_key="k", fpga_id=4)
    app._programmed = True

    resp_404 = requests.Response()
    resp_404.status_code = 404
    http_err_404 = requests.HTTPError(response=resp_404)

    with (
        patch.object(_client, "release_session", side_effect=http_err_404),
        patch.object(
            _client, "reset_fpga", return_value={"job_id": "reset-job-404"}
        ) as mock_reset,
    ):
        job_id = app.release()
        mock_reset.assert_called_once_with(4, "k", app.api_url)
        assert job_id == "reset-job-404"
    assert not app._programmed


def test_release_shields_sigint():
    app = mrg.App("test_app", design="d.py", api_key="k", fpga_id=1)
    app._programmed = True

    sig_during_call = None

    def fake_release(fpga_id, api_key, api_url):
        nonlocal sig_during_call
        sig_during_call = signal.getsignal(signal.SIGINT)
        return "job-1"

    with patch.object(_client, "release_session", side_effect=fake_release):
        app.release()

    assert sig_during_call == signal.SIG_IGN
    # Restored after release()
    assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
