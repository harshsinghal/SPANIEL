# Runtime configuration is part of the model

**Harsh Singhal · 21 October 2026.** The first credential/context run was invalidated after a Qwen3 configuration compatibility fault was found. Its adapter is not a release candidate. A corrected GPU.ai run restarts from the original base with the same data and training recipe.

## What failed

The pinned v2 checkpoint was saved in a newer Transformers format. Its [configuration](https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2/blob/a58063469be8c7b929016d944d8d6d1545bc70a0/config.json) specifies `rope_parameters.rope_theta = 1000000`. RoPE controls how position enters the attention calculation; it is part of the model's computation.

The experiment and source demo used Transformers 4.57.6. Its [Qwen3 configuration class](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/configuration_qwen3.py) reads the older top-level `rope_theta` field, whose default is `10000`. It retained the newer dictionary as an attribute but did not use it to construct the rotary frequencies. Weight loading succeeded without establishing semantic configuration compatibility.

Both the first fine-tune and its baseline/candidate evaluations therefore used the wrong positional setting. A paired comparison under the same wrong setting does not establish performance for the intended base model. The first run's apparent gains cannot be attributed only to better credential/context training data: adaptation to a misconfigured base is a confound.

This was discovered when the fresh PII-retention evaluation diverged from archived v2 predictions. Rescoring all 300 archived constrained predictions with the current source gold and scorer reproduced **0.929767 exact typed span F1** (1,999 true positives, 131 false positives, 171 false negatives). The root and `final/` model-weight files also had identical SHA-256 hashes. The archived score was reproducible from its predictions; the runtime configuration required investigation.

## Repair and verification

The shared [model loader](../model_compat.py) translates this supported default-Qwen3 RoPE configuration for older Transformers versions. It refuses unsupported scaling layouts and conflicting values rather than guessing. Native newer loaders retain their configuration format. The source server, constrained evaluation command, contextual trainer and paired evaluator use this loader; the Docker build includes it.

The published v2 [compatibility revision `fc31a21`](https://huggingface.co/Harsh/qwen3-0.6b-pii-sft-v2/commit/fc31a21a2ac54850ba3c12781ae433413691189d) also adds equivalent legacy fields to the root and `final/` configuration files. Both model-weight hashes remain unchanged. Direct configuration loading was checked with Transformers 4.57.6 and 5.8.0: both construct the intended rotary-frequency tensor. The Docker definition now pins this compatible revision. The contextual training experiment retains its original base revision and applies the verified loader, preserving its recorded provenance.

Verification checks the **actual rotary-frequency tensor** against the tensor implied by theta `1000000`, in addition to unit tests for the schema transition, conflicting values and unsupported layouts. New training manifests and evaluation reports record effective RoPE values and the compatibility-helper hash. The comparison refuses missing or incorrect effective RoPE, even when both reports share the same mistake. `/api/model` exposes the effective setting for application diagnostics.

To repeat the configuration check without downloading model weights, run this from the repository with the experiment dependencies installed:

```python
import torch
from huggingface_hub import snapshot_download
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from model_compat import load_model_config, effective_rope

path = snapshot_download(
    "Harsh/qwen3-0.6b-pii-sft-v2",
    revision="a58063469be8c7b929016d944d8d6d1545bc70a0",
    allow_patterns=["config.json"],
)
config = load_model_config(path)
rotary = Qwen3RotaryEmbedding(config)
expected = 1 / (1000000 ** (
    torch.arange(0, config.head_dim, 2).float() / config.head_dim
))
assert torch.equal(rotary.inv_freq.cpu(), expected)
print(effective_rope(config))
```

This verifies the supported default-RoPE interpretation. It does not verify model weights, detection accuracy or every possible future Qwen configuration.

Inference explicitly records `use_cache=True`. The saved model configuration has `use_cache=False`, but its generation configuration already defaults to caching; this investigation does not claim a caching speedup. Training continues to disable caching while using gradient checkpointing.

## How the experiment was restarted

- The original v2 weights, frozen source texts and labels, replay sample, scenario-family splits, LoRA rank, learning rate, seed and effective batch remain unchanged.
- Training starts again from the original base, with two epochs and the same minimum-development-loss checkpoint rule. It does not continue from the invalid adapter.
- Both baseline and candidate evaluations and the application smoke checks are rerun using the repaired loader.
- Some reserved-test predictions had already been observed under the invalid runtime. No data, labels, thresholds, hyperparameters or selection rules were changed in response. This is a disclosed runtime-repair rerun, not an untouched multi-run benchmark.
- The independent GLiNER baseline uses a separate model/runtime and is unaffected.

The [repair protocol](../experiments/credential-context-runtime-repair.json) records the decision before the corrected restart. The [invalid-run record](../reports/credential-context/invalid-rope-run/INVALIDATED.json) preserves its status and artifact hash; the model reports in that directory are explicitly marked invalid. Its weights were staged privately and withheld from public release. Corrected results belong in the [milestone record](milestones/2026-10-21-credential-context.md).

This failure is part of the research record. A checkpoint hash alone does not establish that the intended model computation was loaded. Runtime configuration and measured regression behavior must accompany the weights when making accuracy claims.
