import subprocess
import sys
from pathlib import Path


PREDICTION_EXAMPLES = [
    "Can we review the project deadline tomorrow?",
    "Are you coming home tonight?",
    "Free prize claim now",
    "Discount voucher offer available today",
    "Client meeting scheduled tomorrow",
]


def check_requirements():
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    missing = []

    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        package = line.strip()
        if not package or package.startswith("#"):
            continue
        module_name = package.split("==")[0].split(">=")[0].replace("-", "_")
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required libraries:")
        for package in missing:
            print(f"- {package}")
        print("\nInstall them with:")
        print(f"{sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)


def run_command(command):
    print("\n" + "=" * 80)
    print("Running: " + " ".join(command))
    print("=" * 80)

    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print("\nCommand failed: " + " ".join(command))
        sys.exit(result.returncode)


def main():
    check_requirements()
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    run_command([sys.executable, "train_email_classifier.py"])

    for message in PREDICTION_EXAMPLES:
        run_command([sys.executable, "predict.py", message])

    run_command(
        [
            sys.executable,
            "inspect_forward.py",
            "Can we review the project deadline tomorrow?",
        ]
    )
    run_command(
        [
            sys.executable,
            "inspect_training_step.py",
            "Can we review the project deadline tomorrow?",
            "work",
        ]
    )

    print("\n" + "=" * 80)
    print("All Phase 1 checks completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
