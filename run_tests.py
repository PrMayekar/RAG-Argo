# run_tests.py
import subprocess
import sys

tests = ["test_etl.py", "test_rag.py", "test_llm.py"]

for t in tests:
    print(f"\n{'#'*40}")
    print(f"Running {t}")
    print('#'*40)
    result = subprocess.run([sys.executable, t], capture_output=False)

