import subprocess
import sys


def main():
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
    ]

    for command in commands:
        print("\n" + "=" * 80)
        print("Running: " + " ".join(command))
        print("=" * 80)
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            sys.exit(result.returncode)

    print("\n" + "=" * 80)
    print("All model API checks completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
