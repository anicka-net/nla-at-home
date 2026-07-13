import ast
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "07_validate_one_axis.ipynb"
ARTIFACTS = REPO / "notebooks" / "artifacts"


def notebook_code():
    data = json.loads(NOTEBOOK.read_text())
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell["cell_type"] == "code"
        and not "".join(cell.get("source", [])).lstrip().startswith(("%", "!"))
    )


def function_source(name):
    source = notebook_code()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_notebook07_injected_generation_has_attention_mask():
    assert "attention_mask=attn" in function_source("describe")


def test_notebook07_moves_injected_vector_to_model_device():
    source = function_source("describe")
    assert "device=embeds.device" in source
    assert "dtype=embeds.dtype" in source


def test_notebook07_uses_plus_one_monte_carlo_estimate():
    source = notebook_code()
    assert "(exceedances + 1) / (len(random_max) + 1)" in source
    assert 'f"< {1/5000:.4f}"' not in source


def test_notebook07_fresh_probes_compare_ordering_without_cherry_picking():
    source = notebook_code()
    assert 'min(probe_projections["pleasant"])' in source
    assert 'max(probe_projections["unpleasant"])' in source
    assert "states[pick]" not in source


def test_notebook07_matches_every_instrument_to_block_19():
    data = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "resid_post_layer_19" in markdown
    assert "source layer 19" in markdown
    assert 'LAYER = 19' in notebook_code()


def test_notebook07_artifacts_match_layer_and_baseline():
    axis = torch.load(
        ARTIFACTS / "qwen25_valence_block19.pt",
        map_location="cpu",
        weights_only=True,
    )
    sae = json.loads(
        (ARTIFACTS / "qwen25_valence_block19_sae.json").read_text()
    )

    assert axis["block_index"] == sae["block_index"] == 19
    assert axis["direction"].shape == (3584,)
    assert torch.allclose(axis["direction"].norm(), torch.tensor(1.0))
    assert axis["pleasant_centroid"].shape == axis["unpleasant_centroid"].shape == (3584,)
    centroid_direction = axis["pleasant_centroid"] - axis["unpleasant_centroid"]
    centroid_direction /= centroid_direction.norm()
    assert torch.allclose(axis["direction"], centroid_direction, atol=1e-5)
    assert sae["top50_span_capture"]["axis"] > sae["top50_span_capture"]["random_p95"]
