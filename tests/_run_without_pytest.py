"""
Not part of the delivered test suite — a throwaway runner used ONLY because
pytest isn't installed in the sandbox this code was authored in. Your
teammates should just run `pytest` (it's in requirements.txt). This file can
be deleted; it's kept only for transparency about how the suite was verified.
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["test_intent", "test_normalization", "test_ranking", "test_service"]

passed = failed = 0
for mod_name in MODULES:
    mod = importlib.import_module(mod_name)
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            try:
                fn()
                print(f"PASS  {mod_name}.{name}")
                passed += 1
            except Exception:
                print(f"FAIL  {mod_name}.{name}")
                traceback.print_exc()
                failed += 1

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
