import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import brain_in_jar_phi4
import brain_in_jar_qwen
import generate_corpus
import generation_utils
import nla_lib
import quickstart
import train_universal_ar


class CharTokenizer:
    injection_id = 999

    def encode(self, text, add_special_tokens=False):
        return [self.injection_id if c == "㈎" else ord(c) for c in text]


def test_ar_truncation_preserves_injection_token():
    examples = [{
        "depth_pct": 71,
        "description": "x" * 1000,
        "activation": torch.zeros(4),
    }]
    dataset = train_universal_ar.LayerARDataset(
        examples, CharTokenizer(), "㈎", max_length=80)
    item = dataset[0]
    assert len(item["input_ids"]) == 80
    assert item["input_ids"][item["inject_pos"]].item() == 999
    assert item["inject_pos"] == len(item["input_ids"]) - 1


def test_lora_reward_rejects_untrained_shallow_layer():
    reward = nla_lib.LoraSLReward(
        model=None, tokenizer=None, n_layers=4, injection_char="㈎",
        min_layer=2)
    with pytest.raises(ValueError, match="trained only for layers >= 2"):
        reward.reconstruct(["x"], [1], "cpu")


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [999 if c == "㈎" else 1 for c in text]

    def __call__(self, prompts, **kwargs):
        return FakeEncoding(input_ids=torch.ones((len(prompts), 1), dtype=torch.long))


class FakeModel:
    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        hidden = [torch.zeros(batch, 1, 3), torch.ones(batch, 1, 3)]
        return types.SimpleNamespace(hidden_states=hidden)


def test_lora_reward_adds_mean_back():
    reward = nla_lib.LoraSLReward(
        FakeModel(), FakeTokenizer(), n_layers=1, injection_char="㈎",
        layer_means={0: torch.full((3,), 2.0)})
    recon = reward.reconstruct(["x"], [0], "cpu")[0]
    assert torch.equal(recon, torch.full((1, 3), 3.0))


def test_decode_generated_handles_both_transformers_shapes():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(int(x)) for x in ids)

    prompt = [1, 2, 3]
    prefixed = types.SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 7, 8]]))
    generated_only = types.SimpleNamespace(sequences=torch.tensor([[7, 8, 9, 10]]))
    assert generation_utils.decode_generated(
        prefixed, prompt, Tok(), stop_text=None) == "7,8"
    assert generation_utils.decode_generated(
        generated_only, prompt, Tok(), stop_text=None) == "7,8,9,10"


def test_qwen_legacy_interface_uses_checkpoint_bytes_and_scaling():
    meta = {
        "prompt_templates": {
            "av": "Here is the vector: <concept>{injection_char}</concept>"
        },
        "training": {},
    }
    prompt, mode = brain_in_jar_qwen.resolve_av_interface(
        meta, "㈎", depth_pct=71)
    assert prompt == "Here is the vector: <concept>㈎</concept>"
    assert mode == "multiply"


def test_qwen_universal_interface_uses_chat_and_depth():
    meta = {
        "prompt_templates": {
            "av": "depth {depth_pct}: {injection_char}"
        },
        "training": {"injection_mode": "normalize", "chat_template": True},
    }
    prompt, mode = brain_in_jar_qwen.resolve_av_interface(
        meta, "㈎", depth_pct=71)
    assert prompt == "depth 71: ㈎"
    assert mode == "normalize"


def test_qwen_ar_prompt_replaces_depth(monkeypatch):
    captured = {}

    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            captured["prompt"] = prompt
            return [1]

    class Model:
        def __call__(self, **kwargs):
            hidden = [torch.zeros(1, 1, 2) for _ in range(22)]
            return types.SimpleNamespace(hidden_states=hidden)

    brain_in_jar_qwen.ar_score(
        Model(), Tokenizer(),
        "depth {depth_pct}: {explanation} {injection_char}",
        "desc", torch.zeros(2), 71, "cpu")
    assert captured["prompt"] == "depth 71: desc ㈎"


def test_phi_confidence_is_centered():
    mean = torch.tensor([10.0, 0.0])
    actual = torch.tensor([11.0, 0.0])
    reconstructed = torch.tensor([9.0, 0.0])
    raw = torch.nn.functional.cosine_similarity(
        actual.unsqueeze(0), reconstructed.unsqueeze(0)).item()
    centered = brain_in_jar_phi4.centered_cosine(
        reconstructed, actual, mean)
    assert raw > 0.99
    assert centered == pytest.approx(-1.0)


def test_generate_category_failure_writes_nothing(tmp_path, monkeypatch):
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("offline")

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions()))
    cat = {
        "id": "A00_test", "group": "A", "count": 1,
        "preamble": "test", "batches": [{"instruction": "one"}],
    }
    monkeypatch.setattr(generate_corpus, "GENERATED_DIR", tmp_path)
    monkeypatch.setattr(generate_corpus.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="refusing to write"):
        generate_corpus.generate_category(client, cat)
    assert not (tmp_path / "A00_test.json").exists()


def test_generate_category_wrong_count_writes_nothing(tmp_path, monkeypatch):
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content='["only one"]'))])
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kwargs: response)))
    cat = {
        "id": "A00_test", "group": "A", "count": 2,
        "preamble": "test", "batches": [{"instruction": "two"}],
    }
    monkeypatch.setattr(generate_corpus, "GENERATED_DIR", tmp_path)
    with pytest.raises(ValueError, match="expected 2 texts"):
        generate_corpus.generate_category(client, cat)
    assert not (tmp_path / "A00_test.json").exists()


def test_quickstart_writes_clean_files(tmp_path, monkeypatch):
    source = tmp_path / "corpus_v2.jsonl"
    source.write_text(json.dumps({
        "id": "A01_000", "text": "hello", "category": "A01_code",
        "group": "A", "layer_pct": 10, "description": "greeting",
    }) + "\n")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(source)))
    monkeypatch.setattr(quickstart, "GENERATED_DIR", tmp_path / "generated")
    quickstart.download_corpus()
    out = tmp_path / "generated" / "descriptions_L10pct_twin_clean.json"
    assert json.loads(out.read_text())[0]["description"] == "greeting"
    assert not list((tmp_path / "generated").glob("*_merged.json"))


def test_quickstart_training_passes_clean_suffix(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        quickstart.subprocess, "run",
        lambda args, check: calls.append(args))
    quickstart.train("gemma3-1b", str(tmp_path / "acts.pt"), "cpu")
    assert "--desc-suffix" in calls[0] and "_twin_clean" in calls[0]
    assert "--strict" in calls[0]
    assert "--desc-suffix" in calls[1] and "_twin_clean" in calls[1]
    assert "--strict" in calls[1]


def test_ar_strict_mode_does_not_fall_back(tmp_path, monkeypatch):
    clean = tmp_path / "descriptions_L4pct_twin_clean.json"
    clean.write_text(json.dumps([{"id": "x", "description": "clean"}]))
    merged = tmp_path / "descriptions_L10pct_merged.json"
    merged.write_text(json.dumps([{"id": "y", "description": "dirty"}]))
    monkeypatch.setattr(train_universal_ar, "GENERATED_DIR", tmp_path)
    descs = train_universal_ar.load_descriptions(
        "_twin_clean", strict=True)
    assert descs == {4: {"x": "clean"}}


@pytest.mark.parametrize("path", [
    REPO / "scripts" / "train_universal_av.py",
    REPO / "scripts" / "train_universal_ar.py",
    REPO / "scripts" / "extract_activations.py",
])
def test_core_device_loaders_do_not_hardcode_gpu_zero_or_auto(path):
    text = path.read_text()
    assert 'device_map={"": 0}' not in text
    assert 'device_map="auto"' not in text


def test_universal_ar_uses_registry_trust_and_writes_prompt_metadata():
    text = (REPO / "scripts" / "train_universal_ar.py").read_text()
    assert "trust_remote_code=True" not in text
    assert '"prompt_templates": {' in text
    assert '"ar": AR_TEMPLATE_DEPTH_SL' in text
