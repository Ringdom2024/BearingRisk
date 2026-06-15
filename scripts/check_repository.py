from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_SIZE = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".pyc"}
FORBIDDEN_PARTS = {"data", "Datasets_dir", "checkpoint", "__pycache__"}
SECRET_PATTERN = re.compile(
    r"(api[_-]?key|password|client[_-]?secret|access[_-]?token)",
    re.IGNORECASE,
)


def main():
    errors = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden directory: {relative}")
        if path.stat().st_size > MAX_PUBLIC_FILE_SIZE:
            errors.append(f"file exceeds 5 MiB: {relative}")
        if (
            path != Path(__file__).resolve()
            and path.suffix.lower()
            in {".py", ".md", ".txt", ".yml", ".yaml", ".cff"}
        ):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if SECRET_PATTERN.search(text):
                errors.append(f"possible secret marker: {relative}")

    required = [
        "README.md",
        "LICENSE",
        "NOTICE.md",
        "CITATION.cff",
        "requirements.txt",
        "robust_tfn/learned_risk.py",
        "results/extended_robustness_paired_tests.csv",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    if errors:
        print("\n".join(errors))
        return 1
    total = sum(path.stat().st_size for path in files)
    print(f"Repository check passed: {len(files)} files, {total} bytes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
