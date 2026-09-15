import subprocess
import sys


def run_step(name, command):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nERROR: {name} failed.")
        sys.exit(result.returncode)


def main():
    python = sys.executable

    run_step(
        "1. Building ML features",
        [python, "build_features.py", "--data", "data", "--out", "ml_training.csv"]
    )

    run_step(
        "2. Training Random Forest",
        [python, "train_random_forest.py"]
    )

    run_step(
        "3. Generating predictions",
        [
            python,
            "predict.py",
            "--data", "data",
            "--model", "model_rf.pkl",
            "--out", "predictions.csv"
        ]
    )

    run_step(
        "4. Validating submission",
        [python, "validate_submission.py", "predictions.csv"]
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Final output: predictions.csv")


if __name__ == "__main__":
    main()