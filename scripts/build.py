import os
import sys
from pathlib import Path

try:
    import PyInstaller.__main__
except ImportError:
    print("PyInstaller not installed. Run: pip install pyinstaller")
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_DIR / "dist"
SPEC_DIR = PROJECT_DIR / "build"

os.chdir(PROJECT_DIR)

excludes = [
    "--exclude-module=torch",
    "--exclude-module=torchvision",
    "--exclude-module=torchaudio",
    "--exclude-module=scipy",
    "--exclude-module=sympy",
    "--exclude-module=pandas",
    "--exclude-module=matplotlib",
    "--exclude-module=seaborn",
    "--exclude-module=pyarrow",
    "--exclude-module=sqlalchemy",
    "--exclude-module=rich",
    "--exclude-module=pydantic",
    "--exclude-module=pytest",
    "--exclude-module=setuptools",
    "--exclude-module=pkg_resources",
    "--exclude-module=requests",
    "--exclude-module=uvicorn",
    "--exclude-module=fastapi",
    "--exclude-module=streamlit",
    "--exclude-module=tensorflow",
    "--exclude-module=transformers",
    "--exclude-module=keras",
]

cmd = [
    "app.py",
    "--name=DeepXpress",
    "--noconsole",
    "--onefile",
    f"--distpath={DIST_DIR}",
    f"--specpath={SPEC_DIR}",
    "--clean",
    "--noconfirm",
] + excludes

print("Running PyInstaller...")
print(f"pyinstaller {' '.join(cmd)}")
print("=" * 60)

PyInstaller.__main__.run(cmd)

print("=" * 60)
print(f"\nBuild complete! .exe at: {DIST_DIR / 'DeepXpress.exe'}")
print("On first run, the .exe will auto-download models to its directory.")
