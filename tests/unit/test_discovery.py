import io
import json
import zipfile

import pytest

from manhattan_reasoning_gym._discovery import discover
from manhattan_reasoning_gym._project import source_bundle


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_python_transitive_packages_without_executing(tmp_path):
    main = put(
        tmp_path,
        "design.py",
        'from helpers import logic\nraise RuntimeError("never execute")',
    )
    put(tmp_path, "helpers/__init__.py", "")
    put(tmp_path, "helpers/logic.py", "from .values import VALUE\nimport math")
    put(tmp_path, "helpers/values.py", "VALUE = 42")
    put(tmp_path, "unrelated.py", 'SECRET = "unrelated"')
    project = discover(main)
    assert set(project.files) == {
        "design.py",
        "helpers/__init__.py",
        "helpers/logic.py",
        "helpers/values.py",
    }


def test_verilog_transitive_parameterized_modules_headers_and_memory(tmp_path):
    main = put(tmp_path, "top.v", "module top; child #(.N((2+2))) u[1:0](); endmodule")
    put(
        tmp_path,
        "rtl/child.v",
        '`include "constants.vh"\nmodule child; grandchild x();'
        ' initial $readmemh("data/a.hex", mem); endmodule',
    )
    put(tmp_path, "rtl/grandchild.v", "module grandchild; endmodule")
    put(tmp_path, "inc/constants.vh", "`define VALUE 1")
    put(tmp_path, "data/a.hex", "1234")
    put(tmp_path, "unused.v", "module unused; endmodule")
    project = discover(main)
    assert set(project.files) == {
        "top.v",
        "rtl/child.v",
        "rtl/grandchild.v",
        "inc/constants.vh",
        "data/a.hex",
    }
    assert project.verilog_sources == ["rtl/child.v", "rtl/grandchild.v"]


def test_textually_included_v_is_not_compiled_twice(tmp_path):
    main = put(
        tmp_path, "top.v", '`include "child.v"\nmodule top; child u(); endmodule'
    )
    put(tmp_path, "child.v", "module child; endmodule")
    project = discover(main)
    assert set(project.files) == {"top.v", "child.v"}
    assert project.verilog_sources == []
    with zipfile.ZipFile(io.BytesIO(source_bundle(main))) as archive:
        assert json.loads(archive.read("mrg-project.json"))["verilog_sources"] == []


def test_directory_can_be_outside_main_folder(tmp_path):
    main = put(tmp_path, "app/top.py", "import helper")
    put(tmp_path, "support/helper.py", "VALUE = 42")
    put(tmp_path, "support/data.bin", "not a supported source")
    project = discover(main, include_dir="../support")
    assert set(project.files) == {"top.py", "helper.py"}


def test_main_nested_in_explicit_project_root(tmp_path):
    main = put(tmp_path, "rtl/top.v", "module top; endmodule")
    put(tmp_path, "include/defs.vh", "`define VALUE 42")
    project = discover(main, include_dir="..")
    assert project.entrypoint == "rtl/top.v"
    assert set(project.files) == {"rtl/top.v", "include/defs.vh"}


def test_explicit_sources_override_discovery(tmp_path):
    main = put(tmp_path, "top.py", "import unrelated")
    put(tmp_path, "unrelated.py", "VALUE = 1")
    put(tmp_path, "chosen.py", "VALUE = 2")
    assert set(discover(main, sources=["chosen.py"]).files) == {"top.py", "chosen.py"}
    assert set(discover(main, sources=[]).files) == {"top.py"}


def test_ambiguous_module_fails_with_actionable_error(tmp_path):
    main = put(tmp_path, "top.v", "module top; child u(); endmodule")
    put(tmp_path, "a.v", "module child; endmodule")
    put(tmp_path, "b.v", "module child; endmodule")
    with pytest.raises(ValueError, match="Ambiguous module"):
        discover(main)


def test_cycles_terminate(tmp_path):
    main = put(tmp_path, "top.py", "import child")
    put(tmp_path, "child.py", "import top")
    assert len(discover(main).files) == 2


def test_comments_do_not_create_dependencies(tmp_path):
    main = put(
        tmp_path,
        "top.v",
        '// `include "missing.vh"\n/* unused u(); */\nmodule top; endmodule',
    )
    put(tmp_path, "unused.v", "module unused; endmodule")
    assert set(discover(main).files) == {"top.v"}


def test_include_cannot_escape_root(tmp_path):
    main = put(tmp_path, "app/top.v", '`include "../secret.vh"')
    put(tmp_path, "secret.vh", "secret")
    with pytest.raises(ValueError, match="escapes"):
        discover(main)


def test_source_symlink_rejected(tmp_path):
    main = put(tmp_path, "top.py", "import helper")
    target = put(tmp_path, "real.py", "VALUE = 1")
    (tmp_path / "helper.py").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        discover(main)


def test_directory_collision_is_not_silently_overwritten(tmp_path):
    main = put(tmp_path, "app/top.py", "import helper")
    put(tmp_path, "support/top.py", "different source")
    with pytest.raises(ValueError, match="Duplicate"):
        discover(main, include_dir="../support")


def test_directory_textual_v_in_subfolder_is_not_compiled_twice(tmp_path):
    main = put(
        tmp_path, "top.v", '`include "child.v"\nmodule top; child u(); endmodule'
    )
    put(tmp_path, "rtl/child.v", "module child; endmodule")
    assert discover(main, include_dir=".").verilog_sources == []


def test_module_name_in_string_does_not_create_ambiguity(tmp_path):
    main = put(tmp_path, "top.v", "module top; child u(); endmodule")
    put(tmp_path, "child.v", "module child; endmodule")
    put(
        tmp_path, "other.v", 'module other; initial $display("module child"); endmodule'
    )
    assert set(discover(main).files) == {"top.v", "child.v"}


def test_macro_include_requests_explicit_directory(tmp_path):
    main = put(tmp_path, "top.v", '`define HEADER "defs.vh"\n`include `HEADER')
    put(tmp_path, "defs.vh", "`define VALUE 42")
    with pytest.raises(ValueError, match="Macro-based include"):
        discover(main)
    assert set(discover(main, include_dir=".").files) == {"top.v", "defs.vh"}
