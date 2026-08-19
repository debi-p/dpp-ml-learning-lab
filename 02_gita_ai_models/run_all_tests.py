import subprocess
import sys


def run(command):
    print("$ " + " ".join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    run([sys.executable, "-m", "unittest", "tests.test_gita_search_assistant"])
    run([sys.executable, "-m", "unittest", "tests.test_gita_embedding_model"])
    run([sys.executable, "-m", "unittest", "tests.test_gita_rag_assistant"])
    run([sys.executable, "-m", "unittest", "tests.test_gita_tiny_transformer"])


if __name__ == "__main__":
    main()
