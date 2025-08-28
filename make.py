#!/usr/bin/env python3
"""
Compatible with Windows, Linux (bash, zsh, fish)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import urllib.request
import tarfile


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_colored(message, color=Colors.OKGREEN):
    """Use different color for different message types"""
    print(f"{color}{message}{Colors.ENDC}")


def run_command(cmd, shell=None):
    """A better wrapping for shell commands"""
    if shell is None:
        shell = platform.system() == "Windows"

    try:
        result = subprocess.run(
            cmd, shell=shell, check=True, capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running command: {cmd}", Colors.FAIL)
        print_colored(f"Error: {e.stderr}", Colors.FAIL)
        return False


def get_python_executable():
    if platform.system() == "Windows":
        return "python"
    else:
        if shutil.which("python3"):
            return "python3"
        return "python"


def create_directories():
    dirs = [
        "src",
        "model",
        "input",
        "info",
    ]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print_colored(f"Created directory: {dir_name}")


def setup_venv():
    python_exe = get_python_executable()
    venv_name = ".venv"

    if not Path(venv_name).exists():
        print_colored("Creating virtual environment...")
        if not run_command([python_exe, "-m", "venv", venv_name]):
            return False
        print_colored("Virtual environment created")
    else:
        print_colored("Virtual environment already exists")

    return True


def get_venv_python():
    if platform.system() == "Windows":
        return Path(".venv/Scripts/python.exe")
    else:
        return Path(".venv/bin/python")


def get_venv_pip():
    if platform.system() == "Windows":
        return Path(".venv/Scripts/pip.exe")
    else:
        return Path(".venv/bin/pip")


def install_requirements():
    if not Path("requirements.txt").exists():
        print_colored(
            "requirements.txt not found, skipping installation", Colors.WARNING
        )
        return True

    pip_exe = get_venv_pip()
    print_colored("Installing requirements...")

    if not run_command([str(pip_exe), "install", "-r", "requirements.txt"]):
        return False

    print_colored("Requirements installed")
    return True


def download_dataset():
    """Download and extract the empathetic dialogues dataset"""
    print_colored("Downloading dataset...")

    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)

    dataset_url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    tar_path = input_dir / "empatheticdialogues.tar.gz"

    try:
        # Download the file
        print_colored("Downloading empatheticdialogues.tar.gz...")
        urllib.request.urlretrieve(dataset_url, tar_path)

        # Extract the tar file
        print_colored("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(input_dir)

        # Remove the tar file
        tar_path.unlink()
        print_colored("Dataset downloaded and extracted successfully!")
        return True

    except Exception as e:
        print_colored(f"Error downloading dataset: {str(e)}", Colors.FAIL)
        return False


def clean():
    print_colored("Cleaning build artifacts...")

    patterns_to_remove = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        "*.egg-info",
        "build",
        "dist",
        ".coverage",
    ]

    for pattern in patterns_to_remove:
        if platform.system() == "Windows":
            run_command(f"del /s /q {pattern}", shell=True)
            run_command(f"rmdir /s /q {pattern}", shell=True)
        else:
            run_command(f"find . -name '{pattern}' -exec rm -rf {{}} +", shell=True)

    print_colored("Cleaned build artifacts")


def show_help():
    """Show help message"""
    help_text = f"""
{Colors.HEADER}Project Makefile{Colors.ENDC}

{Colors.BOLD}Available commands:{Colors.ENDC}
  {Colors.OKCYAN}setup{Colors.ENDC}      - Setup project (create dirs, venv, install deps)
  {Colors.OKCYAN}install{Colors.ENDC}    - Install requirements in virtual environment
  {Colors.OKCYAN}dataset{Colors.ENDC}    - Download empathetic dialogues dataset
  {Colors.OKCYAN}clean{Colors.ENDC}      - Clean build artifacts and cache files
  {Colors.OKCYAN}help{Colors.ENDC}       - Show this help message

{Colors.BOLD}Usage:{Colors.ENDC}
  python make.py <command>

{Colors.BOLD}Examples:{Colors.ENDC}
  python make.py setup     # Full project setup
  python make.py dataset   # Download dataset
  python make.py clean     # Clean artifacts
"""
    print(help_text)


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "setup":
        print_colored(
            "Setting up Transformer based therapist project...", Colors.HEADER
        )
        create_directories()
        if setup_venv() and install_requirements():
            print_colored("Project setup complete!", Colors.OKGREEN)

        else:
            print_colored("Setup failed", Colors.FAIL)
            sys.exit(1)

    elif command == "install":
        if not Path(".venv").exists():
            print_colored(
                "Virtual environment not found. Run 'python make.py setup' first.",
                Colors.FAIL,
            )
            sys.exit(1)
        install_requirements()

    elif command == "dataset":
        if not download_dataset():
            sys.exit(1)

    elif command == "clean":
        clean()

    elif command in ["help", "-h", "--help"]:
        show_help()

    else:
        print_colored(f"Unknown command: {command}", Colors.FAIL)
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
