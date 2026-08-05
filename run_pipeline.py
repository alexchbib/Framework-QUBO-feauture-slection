import os
import sys
import argparse
import subprocess

def run_script(script_path, cwd=None):
    print(f"\n=================================================================")
    print(f"RUNNING: {script_path}")
    print(f"=================================================================\n")
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(script_path))
    cmd = [sys.executable, os.path.basename(script_path)]
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for ADNI Multi-Task Feature Selection")
    parser.add_argument('--all', action='store_true', help="Run all benchmarks, cost evaluations, and ablation studies")
    parser.add_argument('--bm1', action='store_true', help="Run Benchmark 1 (FISTA MTFL)")
    parser.add_argument('--bm2', action='store_true', help="Run Benchmark 2 (XGBoost)")
    parser.add_argument('--baseline', action='store_true', help="Run Baseline 3 (Greedy Panel Elimination)")
    parser.add_argument('--costs', action='store_true', help="Run Medical Test Cost Evaluator")
    parser.add_argument('--ablation', action='store_true', help="Run Feature Modality Ablation Study")

    args = parser.parse_args()

    # Default to --all if no arguments provided
    if not any([args.all, args.bm1, args.bm2, args.baseline, args.costs, args.ablation]):
        args.all = True

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.all or args.bm1:
        run_script(os.path.join(root_dir, 'benchmark 1 multitask learning', 'mtfl_benchmark.py'))

    if args.all or args.bm2:
        run_script(os.path.join(root_dir, 'benchmark 2 xgboost', 'xgb_benchmark.py'))

    if args.all or args.baseline:
        run_script(os.path.join(root_dir, 'src', 'baselines', 'greedy_panel_baseline.py'), cwd=root_dir)

    if args.all or args.costs:
        run_script(os.path.join(root_dir, 'evaluate_costs.py'), cwd=root_dir)

    if args.all or args.ablation:
        run_script(os.path.join(root_dir, 'src', 'ablation_study.py'), cwd=root_dir)

    print("\n=================================================================")
    print("PIPELINE EXECUTION COMPLETE! ALL RESULTS GENERATED AND SAVED.")
    print("=================================================================\n")

if __name__ == '__main__':
    main()
