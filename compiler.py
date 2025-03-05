import shutil
import site
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import utilities.utils as utils


def run_command(command: list, use_shell: bool = False) -> None:
    subprocess.run(command, check=True, shell=use_shell)


def install_requirements() -> None:
    print("\nInstalling requirements...")
    run_command(['pip', 'install', '-r', 'requirements.txt'])


def install_package(name: str) -> None:
    print(f"\n...Installing package {name}...")
    run_command(["pip", "install", name])


def uninstall_package(name: str) -> None:
    temp_dir = Path(f"{site.getsitepackages()[1]}/~addle")
    if temp_dir.exists():
        print("\nRemoving undeleted temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n...Uninstalling package {name}...")
    run_command(["pip", "uninstall", name])


def download_all_models() -> None:
    from paddleocr.paddleocr import MODEL_URLS, DEFAULT_OCR_MODEL_VERSION, PaddleOCR

    languages = MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys()
    for lang in languages:
        print(f"\nChecking for {lang} language models...")
        utils.Config.ocr_opts["base_dir"] = utils.Config.model_dir
        _ = PaddleOCR(lang=lang, **utils.Config.ocr_opts)


def remove_non_onnx_models() -> None:
    print("\nRemoving all non Onnx Models...")
    for file in utils.Config.model_dir.rglob("*.*"):
        if not file.is_dir() and file.name != "model.onnx":
            print(f"Removing file: {file}")
            file.unlink()


def compile_program() -> None:
    cmd = [
        "nuitka",
        "--standalone",
        "--enable-plugin=tk-inter",
        "--windows-console-mode=disable",
        "--include-package-data=paddleocr",
        "--include-data-files=VSE.ico=VSE.ico",
        "--include-data-dir=models=models",
        "--windows-icon-from-ico=VSE.ico",
        "--remove-output",
        "gui.py"
    ]
    print(f"\nCompiling program with Nuitka... \nCommand: {cmd}")
    run_command(cmd, True)


def rename_exe() -> None:
    print("\nRenaming exe file...")
    exe_file = Path("gui.dist/gui.exe")
    exe_file.rename("gui.dist/VSE.exe")


def zip_files() -> None:
    print("\nZipping distribution files...")
    shutil.make_archive("gui.dist", "zip", "gui.dist")


def delete_dist_dir() -> None:
    print("\nRemoving distribution directory...")
    shutil.rmtree("gui.dist")


def main() -> None:
    start_time = perf_counter()

    install_requirements()
    install_package("Nuitka==2.6.7")
    download_all_models()
    remove_non_onnx_models()
    uninstall_package("paddlepaddle")
    uninstall_package("requests")
    compile_program()
    rename_exe()
    zip_files()
    delete_dist_dir()

    print(f"\nCompilation Duration: {timedelta(seconds=round(perf_counter() - start_time))}")


if __name__ == '__main__':
    main()
