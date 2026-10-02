"""v1.0.24: hatch -o must not pre-create the output directory.

The skill registration yaml must always live at skill_dir/agenthatch.yaml.
The old resolver redirected the yaml into the -o output directory, whose
mkdir at step 11 pre-created the directory before Phase 3 — the generator
then saw an existing output dir and raised FileExistsError, so
``hatch <skill> -o <fresh dir>`` failed on the first run and demanded
``--force``. The resolver now takes no output argument at all: the
compiler enforces that the output directory can never be its target.
"""
from agenthatch.cli.commands.hatch import _resolve_yaml_path


class TestYamlPathResolution:
    def test_yaml_always_in_skill_dir(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        path = _resolve_yaml_path(skill_dir)
        assert path == skill_dir / "agenthatch.yaml"
        assert path.parent == skill_dir
