import io
import json
import zipfile

import pytest

from manhattan_reasoning_gym._project import source_bundle


def test_only_explicit_sources_are_uploaded(tmp_path):
    main = tmp_path / "design.py"
    main.write_text("from helpers import logic")
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers/logic.py").write_text("VALUE = 42")
    (tmp_path / ".env").write_text("do not upload")
    with zipfile.ZipFile(
        io.BytesIO(source_bundle(str(main), ["helpers/logic.py"]))
    ) as archive:
        assert set(archive.namelist()) == {
            "mrg-project.json",
            "design.py",
            "helpers/logic.py",
        }
        assert json.loads(archive.read("mrg-project.json"))["entrypoint"] == "design.py"


@pytest.mark.parametrize(
    "source", ["../other.py", "/other.py", "missing.py", "design.py", ".env"]
)
def test_invalid_source_fails_locally(tmp_path, source):
    main = tmp_path / "design.py"
    main.write_text("")
    with pytest.raises(ValueError):
        source_bundle(str(main), [source])


def test_reject_symlink(tmp_path):
    main = tmp_path / "design.py"
    main.write_text("")
    (tmp_path / "link.py").symlink_to(main)
    with pytest.raises(ValueError, match="symlink"):
        source_bundle(str(main), ["link.py"])


def test_submit_uses_zip_and_preserves_options(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from manhattan_reasoning_gym import _client

    main = tmp_path / "design.v"
    main.write_text("")
    (tmp_path / "child.v").write_text("")
    response = Mock()
    response.json.return_value = {"job_id": "test-job"}
    post = Mock(return_value=response)
    monkeypatch.setattr(_client.requests, "post", post)
    assert (
        _client.submit(
            str(main), "key", "https://example.test", top="main", sources=["child.v"]
        )
        == "test-job"
    )
    args = post.call_args.kwargs
    assert args["data"]["top"] == "main"
    name, content, kind = args["files"]["file"]
    assert name == "project.zip"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert "child.v" in archive.namelist()


def test_app_forwards_sources(monkeypatch):
    from unittest.mock import Mock

    from manhattan_reasoning_gym import _app

    submit = Mock(return_value="job")
    monkeypatch.setattr(_app._client, "submit", submit)
    monkeypatch.setattr(_app._client, "poll_job", Mock(return_value={"fpga_id": 2}))
    app = _app.App("test", design="design.py", sources=["helper.py"], api_key="key")
    app._program()
    assert submit.call_args.kwargs["sources"] == ["helper.py"]


def test_cli_appends_repeated_sources(monkeypatch):
    import sys

    from manhattan_reasoning_gym import _app, _cli

    selected = []

    def load(_):
        app = _app.App("test", design="design.v", sources=["base.v"], api_key="key")
        app._entrypoint = lambda: selected.extend(app.sources)

    monkeypatch.setattr(_cli, "_load_user_module", load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mrg",
            "run",
            "client.py",
            "--no-program",
            "--source",
            "child.v",
            "--source",
            "inc/defs.vh",
        ],
    )
    _cli.main()
    assert selected == ["base.v", "child.v", "inc/defs.vh"]
