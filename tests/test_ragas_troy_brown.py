import os

import pytest


@pytest.mark.skipif(
    not os.getenv("TROY_BROWN_PATH"),
    reason="Set TROY_BROWN_PATH to a Troy Brown deposition file to run RAGAS.",
)
def test_ragas_troy_brown_faithfulness_and_correctness():
    ragas = pytest.importorskip("ragas")

    # This is a harness scaffold. It is intentionally skipped unless the
    # deposition path + provider keys are configured in the environment.
    #
    # Expected usage:
    # - export TROY_BROWN_PATH=/path/to/troy_brown.pdf (or .txt)
    # - export OPENAI_API_KEY=... (or set the provider used by your evaluator)
    # - run: pytest -k ragas

    from ragas.metrics import faithfulness, answer_correctness

    # TODO: Build a Dataset of Q/A pairs for the deposition.
    # TODO: Call the system under test to get answers + contexts.
    # TODO: Evaluate with ragas.evaluate and assert thresholds.

    # Placeholder assertion so the harness is syntactically valid.
    assert faithfulness is not None and answer_correctness is not None
