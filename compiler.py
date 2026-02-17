import platform
import shutil
import site
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter

from infra.app_paths import AppPaths


def run_command(command: list, use_shell: bool = False) -> None:
    subprocess.run(command, check=True, shell=use_shell)


def install_requirements(device: str) -> None:
    print(f"\nInstalling {device} requirements...")
    run_command(['pip', 'install', '-r', f'requirements-{device}.txt'])


def uninstall_requirements(device: str) -> None:
    print(f"\nUninstalling {device} requirements...")
    run_command(['pip', 'uninstall', '-r', f'requirements-{device}.txt', '-y'])


def install_package(name: str) -> None:
    print(f"\n...Installing package {name}...")
    run_command(["pip", "install", name])


def download_all_models() -> None:
    from shared.config import CONFIG
    from custom_ocr import CustomPaddleOCR

    txt_line_ori_models = ["PP-LCNet_x0_25_textline_ori", "PP-LCNet_x1_0_textline_ori"]
    det_models = ['PP-OCRv5_mobile_det', 'PP-OCRv5_server_det']
    rec_models = ['PP-OCRv5_mobile_rec', 'PP-OCRv5_server_rec', 'arabic_PP-OCRv5_mobile_rec',
                  'cyrillic_PP-OCRv5_mobile_rec', 'devanagari_PP-OCRv5_mobile_rec', 'en_PP-OCRv5_mobile_rec',
                  'eslav_PP-OCRv5_mobile_rec', 'korean_PP-OCRv5_mobile_rec', 'latin_PP-OCRv5_mobile_rec',
                  'ta_PP-OCRv5_mobile_rec', 'te_PP-OCRv5_mobile_rec']
    CONFIG.ocr_opts["use_gpu"] = False
    for model in txt_line_ori_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(textline_orientation_model_name=model, **CONFIG.ocr_opts)
        print("-" * 150)
    print("-" * 200)
    for model in det_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(text_detection_model_name=model, **CONFIG.ocr_opts)
        print("-" * 150)
    print("-" * 200)
    for model in rec_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(text_recognition_model_name=model, **CONFIG.ocr_opts)
        print("-" * 150)


def remove_non_onnx_models() -> None:
    print("\nRemoving all non Onnx Models...")
    for file in AppPaths.models().rglob("*"):
        if file.is_file() and ".onnx" not in file.name and ".yml" not in file.name:
            print(f"Removing file: {file}")
            file.unlink()


def remove_compiler_leftovers() -> None:
    print("\nRemoving compiler leftovers...")
    Path(f"{AppPaths.exe_name}.spec").unlink()
    shutil.rmtree("build")
    shutil.rmtree("dist")


def compile_program(gpu_enabled: bool) -> None:
    cmd = [
        "pyinstaller",
        f"--add-data={AppPaths.icon_file}:{AppPaths.icon_file.parent.name}",
        f"--add-data={AppPaths.version_file}:{AppPaths.version_file.parent.name}",
        f"--add-data={AppPaths.models()}:{AppPaths.models().name}",
        "--collect-data=custom_ocr",
        f"--icon={AppPaths.icon_file}",
        f"--name={AppPaths.exe_name}",
        "--noconsole",
        "--noconfirm",
        "--clean",
    ]
    if gpu_enabled:
        cmd.append("--collect-binaries=nvidia")
    if platform.system() == "Linux":
        cmd.append("--hidden-import=PIL._tkinter_finder")
    cmd.append("gui.py")
    print(f"\nCompiling program with PyInstaller... \nCommand: {' '.join(cmd)}")
    run_command(cmd)


def create_installer(gpu_enabled: bool) -> None:
    version = AppPaths.version_file.read_text()
    name = f"{AppPaths.exe_name}-{platform.system()}-{'GPU' if gpu_enabled else 'CPU'}-v{version}"
    inno_exe = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")
    if platform.system() == "Windows" and inno_exe.exists():
        cmd = [
            str(inno_exe),
            f"/DMyAppVersion={version}",
            f"/DOutputBaseFilename={name}",
            str(AppPaths.inno_installer_file),
        ]
        print(f"\nCreating {platform.system()} installer for program... \nCommand: {' '.join(cmd)}")
        run_command(cmd)
    else:
        print(f"\nZipping distribution files...")
        shutil.make_archive(name, "zip", f"dist/{AppPaths.exe_name}")


def remove_site_pkg_tempdirs() -> None:
    print("\nChecking for site package temporary directories...")
    temp_dir = Path(f"{site.getsitepackages()[1]}")
    if not temp_dir.exists():
        print(f"Site package temporary directory not found: {temp_dir}")
        return
    for folder in temp_dir.iterdir():
        if folder.name.startswith("~"):
            print(f"Deleting temp folder: {folder}")
            shutil.rmtree(folder)


def check_for_gpu() -> bool:
    print("\nChecking for GPU support...")
    gpu_supported = False
    try:
        output = subprocess.check_output("nvidia-smi")
        print(f"NVIDIA GPU detected\n{output.decode()}")
        gpu_supported = True
    except Exception as error:
        print(f"No NVIDIA GPU detected or nvidia-smi not installed. Error: {error}")
    return gpu_supported


def build_dist(gpu_enabled: bool) -> None:
    start_time = perf_counter()
    remove_site_pkg_tempdirs()
    if gpu_enabled:
        if not check_for_gpu(): return
        uninstall_requirements("cpu")
        install_requirements("gpu")
    else:
        uninstall_requirements("gpu")
        install_requirements("cpu")
    install_package("pyinstaller==6.18.0")
    download_all_models()
    remove_non_onnx_models()
    compile_program(gpu_enabled)
    create_installer(gpu_enabled)
    remove_compiler_leftovers()
    print(f"\nCompilation Duration: {timedelta(seconds=round(perf_counter() - start_time))}")


if __name__ == '__main__':
    build_dist(gpu_enabled=False)
    build_dist(gpu_enabled=True)
