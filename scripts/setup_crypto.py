"""Fetch pinned upstream code and build native TFHE locally."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tfhe": ("https://github.com/tfhe/tfhe.git", "9373b5d8a5b022ca5b4e112ae1c2440bc18e160c"),
    "non-interactive-ppfim": ("https://github.com/Airscope/non-interactive-ppfim.git", "d862f5c0d9dd4fa8cd155abdd8bbaa3cc6e63a81"),
}


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    for name, (url, commit) in SOURCES.items():
        target = ROOT / "third_party" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", url, target])
            run(["git", "-C", target, "checkout", "--detach", commit])
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Different upstream revision in {target}; refusing to replace existing files")
    tfhe = ROOT / "third_party/tfhe"
    run(["cmake", "-S", tfhe / "src", "-B", tfhe / "build", "-DCMAKE_BUILD_TYPE=optim",
         "-DENABLE_TESTS=OFF", "-DENABLE_FFTW=OFF", "-DENABLE_SPQLIOS_FMA=ON"])
    run(["cmake", "--build", tfhe / "build", "-j", args.jobs])
    run(["cmake", "-S", ROOT / "experiments/crypto_reproductions", "-B", ROOT / "build/crypto_reproductions"])
    run(["cmake", "--build", ROOT / "build/crypto_reproductions", "-j", args.jobs])


if __name__ == "__main__":
    main()
