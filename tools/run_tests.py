"""Run synthetic tests only, independent of the caller's working directory."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('synthetic_original', ROOT / 'tests/test_pares_ad.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
functions = [f for name, f in sorted(vars(module).items()) if name.startswith('test_') and callable(f)]
for function in functions:
    function()
    print('PASS', function.__name__)
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_release.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('Synthetic tests:', len(functions) + result.testsRun)
raise SystemExit(0 if result.wasSuccessful() else 1)
