"""Preserve stored Qwen3 RoPE settings across Transformers config formats."""
import json
import math
from pathlib import Path


def reconcile_config(config, stored):
    """Translate the newer default-RoPE schema for older Qwen3 loaders.

    Transformers 4.x otherwise retains rope_parameters as an unused attribute
    while silently applying its own rope_theta default. Refuse unsupported
    translations instead of guessing how a checkpoint should be interpreted.
    """
    parameters = stored.get('rope_parameters')
    if parameters is None or not hasattr(config, 'rope_theta'):
        return config
    if stored.get('model_type') != 'qwen3':
        raise ValueError('new RoPE schema requires a compatible model loader')
    if not isinstance(parameters, dict) or parameters.get('rope_type') != 'default':
        raise ValueError('only default Qwen3 RoPE translation is supported')
    if set(parameters) - {'rope_type', 'rope_theta'}:
        raise ValueError('unsupported Qwen3 RoPE parameters')
    theta = parameters.get('rope_theta')
    if (isinstance(theta, bool) or not isinstance(theta, (int, float))
            or not math.isfinite(theta) or theta <= 0):
        raise ValueError('invalid stored RoPE theta')
    if 'rope_theta' in stored and stored['rope_theta'] != theta:
        raise ValueError('conflicting stored RoPE theta values')
    if stored.get('rope_scaling') is not None:
        raise ValueError('conflicting stored RoPE scaling')
    config.rope_theta = theta
    config.rope_scaling = None
    return config


def load_model_config(model_dir):
    from transformers import AutoConfig
    stored = json.loads((Path(model_dir)/'config.json').read_text())
    return reconcile_config(AutoConfig.from_pretrained(model_dir), stored)


def effective_rope(config):
    """Record the parameters actually consumed by this loader."""
    if hasattr(config, 'rope_theta'):
        return {'rope_theta': config.rope_theta,
                'rope_scaling': getattr(config, 'rope_scaling', None)}
    return {'rope_parameters': getattr(config, 'rope_parameters', None)}
