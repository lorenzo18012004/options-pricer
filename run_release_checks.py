"""
Run release checks: compile + quant regression tests.
Usage: python run_release_checks.py
"""

import compileall
import subprocess
import sys


def main():
    print("Compiling Python files...")
    ok = compileall.compile_dir(".", quiet=1, force=False)
    if not ok:
        print("Compilation had warnings.")
    print("Running quant regression tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_quant_regression.py", "-v", "--tb=short"],
        capture_output=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
