"""The public API is organized into build / cloud / sandbox namespaces, with the
old flat names kept as back-compat aliases pointing at the same objects.
"""

from __future__ import annotations

import manhattan_reasoning_gym as mrg


def test_namespaces_exist():
    assert hasattr(mrg, "build")
    assert hasattr(mrg, "cloud")
    assert hasattr(mrg, "sandbox")


def test_namespace_members():
    assert callable(mrg.build.synth) and callable(mrg.build.pnr)
    assert hasattr(mrg.cloud, "App") and callable(mrg.cloud.secret)
    assert callable(mrg.sandbox.promote)


def test_flat_aliases_point_to_namespaced():
    assert mrg.synth is mrg.build.synth
    assert mrg.pnr is mrg.build.pnr
    assert mrg.promote is mrg.sandbox.promote
    assert mrg.App is mrg.cloud.App
    assert mrg.secret is mrg.cloud.secret


def test_sandbox_surface_is_just_promote():
    # A sandboxed agent's namespace must not expose the direct-cloud surface.
    assert set(mrg.sandbox.__all__) == {"promote"}
    assert not hasattr(mrg.sandbox, "App")
