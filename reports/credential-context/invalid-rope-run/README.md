# Invalid run — diagnostic records only

**Do not use these measurements as accuracy claims for the released model.** Transformers 4.57.6 silently used RoPE theta `10000` instead of the checkpoint's stored value `1000000`. Both training and inference were affected. The resulting adapter was not publicly released.

The records preserve what happened before the compatibility error was discovered. Their numerical observations do not establish gains from the synthetic data: adaptation to the incorrectly configured base is a confound. Even the application checks and Apple MPS development run in this directory used that configuration.

See the [investigation](../../../docs/runtime-compatibility.md), [invalidation manifest](INVALIDATED.json) and [repair protocol](../../../experiments/credential-context-runtime-repair.json). Corrected evaluations belong in the parent directory. The GLiNER baseline was run with a separate model and runtime and is unaffected.
