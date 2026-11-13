# Verification scripts

Run with the project's Python (the pipx jupyterlab venv, which holds torch):

```bash
PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
PYTHONPATH=src $PY tests/test_pipeline_endtoend.py /tmp/synthcheck
PYTHONPATH=src $PY tests/test_models_causality.py
PYTHONPATH=src $PY tests/test_split_and_slicer.py /tmp/synthcheck
```

`test_pipeline_endtoend.py` generates synthetic captures with known ground
truth, runs the full preprocessing chain, and asserts anonymisation actually
landed in the stored bytes. `test_models_causality.py` checks the property the
early-exit design depends on: head *k* must produce identical logits from a
truncated and a full input, so reported "bytes read" is honest.

`test_split_and_slicer.py` covers the budget slicer and one bug that nearly
became a published finding: the length-balanced padding control used to derive
its length bins and its retained class set independently for each split, so the
model trained on one label set and was scored against another. That collapses
to chance, which looks exactly like the dramatic result the control exists to
detect. The test asserts every split shares one class set, which is what
`length_balance_plan` guarantees.
