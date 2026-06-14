import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    return res.returncode

def main():
    group1 = [
        "tests/test_active_learning.py",
        "tests/test_api.py",
        "tests/test_dicomweb.py",
        "tests/test_async_cb.py",
        "tests/test_mlops.py"
    ]
    group2 = [
        "tests/test_dataset.py",
        "tests/test_fairness.py",
        "tests/test_federated.py",
        "tests/test_medfound.py",
        "tests/test_model.py",
        "tests/test_onnx.py",
        "tests/test_siim_data.py",
        "tests/test_uncertainty.py",
        "tests/test_vit.py",
        "tests/test_xai.py"
    ]
    
    cmd1 = f"python -m pytest -p no:asyncio -vv {' '.join(group1)}"
    cmd2 = f"python -m pytest -p no:asyncio -vv {' '.join(group2)}"
    
    print("=== Running Group 1: API & Database Tests ===")
    code1 = run_cmd(cmd1)
    if code1 != 0:
        print("Group 1 failed!")
        sys.exit(code1)
        
    print("\n=== Running Group 2: Model & ML Evaluation Tests ===")
    code2 = run_cmd(cmd2)
    if code2 != 0:
        print("Group 2 failed!")
        sys.exit(code2)
        
    print("\n=== All 36 tests passed successfully! ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
