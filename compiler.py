import platform
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
        _ = PaddleOCR(use_gpu=False, lang=lang, **utils.Config.ocr_opts)


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
        "--include-data-files=vsx.ico=vsx.ico",
        "--include-data-dir=models=models",
        "--windows-icon-from-ico=vsx.ico",
        "--remove-output",
        "gui.py"
    ]
    print(f"\nCompiling program with Nuitka... \nCommand: {cmd}")
    run_command(cmd, True)


def rename_exe() -> None:
    print("\nRenaming exe file...")
    exe_file = Path("gui.dist/gui.exe")
    exe_file.rename("gui.dist/VSX.exe")


def get_gpu_files() -> None:
    print("\nCopying GPU files...")
    gpu_files_dir = Path(site.getsitepackages()[1], "nvidia")
    required_dirs = ["cudnn", "cufft", "cublas", "cuda_runtime"]
    if platform.system() == "Linux":
        required_dirs.extend(["cuda_nvrtc", "curand"])
    for dir_name in required_dirs:
        shutil.copytree(gpu_files_dir / dir_name, f"gui.dist/nvidia/{dir_name}")


def zip_files(gpu_enabled: bool) -> None:
    print("\nZipping distribution files...")
    name = f"VSX-{platform.system()}-{'GPU' if gpu_enabled else 'CPU'}-v"
    shutil.make_archive(name, "zip", "gui.dist")


def delete_dist_dir() -> None:
    print("\nRemoving distribution directory...")
    shutil.rmtree("gui.dist")


def main(gpu_enabled: bool = True) -> None:
    start_time = perf_counter()

    if gpu_enabled:
        uninstall_package("onnxruntime")
        install_package("onnxruntime-gpu[cuda,cudnn]==1.21.0")
    else:
        uninstall_package("onnxruntime-gpu")
        install_package("onnxruntime==1.21.0")
    install_requirements()
    install_package("Nuitka==2.6.9")
    download_all_models()
    remove_non_onnx_models()
    uninstall_package("paddlepaddle")
    uninstall_package("requests")
    compile_program()
    rename_exe()
    if gpu_enabled:
        get_gpu_files()
    zip_files(gpu_enabled)
    delete_dist_dir()

    print(f"\nCompilation Duration: {timedelta(seconds=round(perf_counter() - start_time))}")


if __name__ == '__main__':
    main()
