"""
Shared test fixtures.

The important one is `test_catalog`: an autouse fixture that installs a fixed,
provider-agnostic catalog for the whole session. Tests refer to models as
CHEAP / MIDDLE / FRONTIER / JUDGE rather than by literal id, so changing a provider
in models.yaml can never again break the test suite — which is exactly what happened
when the Groq -> Together AI migration left ~45 assertions asserting Groq ids.
"""

import textwrap

import pytest

from smartroute.catalog import load_catalog, set_catalog

# Real ids so prices resolve from litellm's registry, but pinned here so the suite is
# independent of whatever the shipped models.yaml currently points at.
CHEAP_ID = "together_ai/openai/gpt-oss-20b"
MIDDLE_ID = "together_ai/openai/gpt-oss-120b"
FRONTIER_ID = "together_ai/Qwen/Qwen3.5-397B-A17B"
JUDGE_ID = "together_ai/mistralai/Mistral-Small-24B-Instruct-2501"

_FIXTURE = f"""
    version: 1
    models:
      - {{alias: cheap,    id: {CHEAP_ID},    role: cheap,    env_key: TEST_KEY}}
      - {{alias: middle,   id: {MIDDLE_ID},   role: middle,   env_key: TEST_KEY}}
      - {{alias: frontier, id: {FRONTIER_ID}, role: frontier, env_key: TEST_KEY}}
      - {{alias: judge,    id: {JUDGE_ID},    role: judge,    env_key: TEST_KEY}}
    routing:
      tiers: [cheap, middle, frontier]
      default_tier: middle
"""


@pytest.fixture(scope="session")
def catalog_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("catalog") / "models.yaml"
    path.write_text(textwrap.dedent(_FIXTURE))
    return path


@pytest.fixture(autouse=True)
def test_catalog(catalog_path, request):
    """Install the fixture catalog for every test except test_catalog.py's own."""
    if request.node.fspath.basename == "test_catalog.py":
        set_catalog(None)
        yield None
        return
    catalog = load_catalog(catalog_path)
    set_catalog(catalog)
    yield catalog
    set_catalog(None)
