import unittest
from types import SimpleNamespace

from model_compat import reconcile_config, effective_rope


class ModelConfigCompatibilityTests(unittest.TestCase):
    def test_new_qwen_config_cannot_silently_use_old_theta_default(self):
        config = SimpleNamespace(rope_theta=10000, rope_scaling=None)
        stored = {'model_type': 'qwen3', 'rope_parameters':
                  {'rope_type': 'default', 'rope_theta': 1000000}}
        reconcile_config(config, stored)
        self.assertEqual(effective_rope(config),
                         {'rope_theta': 1000000, 'rope_scaling': None})

    def test_legacy_and_native_new_configs_are_preserved(self):
        old = SimpleNamespace(rope_theta=1000000, rope_scaling=None)
        self.assertIs(reconcile_config(old, {'rope_theta': 1000000}), old)
        new = SimpleNamespace(rope_parameters={'rope_type': 'default', 'rope_theta': 1000000})
        self.assertIs(reconcile_config(new, vars(new)), new)

    def test_unsupported_or_conflicting_translation_fails(self):
        for params in ({'rope_type': 'linear', 'rope_theta': 1000000, 'factor': 2},
                       {'rope_type': 'default', 'rope_theta': float('nan')},
                       {'rope_type': 'default', 'rope_theta': True}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                reconcile_config(SimpleNamespace(rope_theta=10000),
                                 {'model_type': 'qwen3', 'rope_parameters': params})
        with self.assertRaises(ValueError):
            reconcile_config(SimpleNamespace(rope_theta=10000),
                             {'model_type': 'qwen3', 'rope_theta': 10000,
                              'rope_parameters': {'rope_type': 'default', 'rope_theta': 1000000}})
