import sys
import time
import pathlib
import asyncio
import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOK_FILES = [
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_model_comparison.ipynb",
    "04_imbalance_and_tuning.ipynb",
    "05_final_evaluation.ipynb",
    "06_call_policy.ipynb",
    "07_segments_and_responsible_ai.ipynb",
    "08_business_interpretation.ipynb",
]

def run_notebook(nb_path: pathlib.Path):
    print(f"\n{'='*70}", flush=True)
    print(f"Executing: {nb_path.name}", flush=True)
    print(f"{'='*70}", flush=True)
    start_time = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(
        nb,
        timeout=1200,  # 20 minutes per cell
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    
    client.execute()
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    elapsed = time.time() - start_time
    print(f"SUCCESS: {nb_path.name} finished in {elapsed:.1f}s", flush=True)
    return elapsed

def main():
    total_start = time.time()
    results = {}
    
    for i, nb_name in enumerate(NOTEBOOK_FILES, 1):
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            print(f"ERROR: Notebook not found: {nb_path}", flush=True)
            sys.exit(1)
            
        print(f"\n[{i}/{len(NOTEBOOK_FILES)}] Starting {nb_name}...", flush=True)
        try:
            elapsed = run_notebook(nb_path)
            results[nb_name] = ("SUCCESS", elapsed)
        except Exception as e:
            print(f"\nFAILED in {nb_name}: {e}", flush=True)
            results[nb_name] = ("FAILED", 0.0)
            sys.exit(1)
            
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}", flush=True)
    print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY!", flush=True)
    print(f"Total time: {total_elapsed:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)
    for name, (status, elap) in results.items():
        print(f"  {name:35s} : {status} ({elap:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
