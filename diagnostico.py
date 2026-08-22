import sys
import shutil
import subprocess
import os

def check(cmd):
    path = shutil.which(cmd)
    if not path:
        return "No instalado"
    try:
        out = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT, text=True).strip()
        return f"{path} ({out.splitlines()[0]})"
    except Exception:
        return path

print("=" * 60)
print(" DIAGNÓSTICO DEL SISTEMA (macOS) ")
print("=" * 60)
print(f"Python ejecutor actual : {sys.executable}")
print(f"Versión de Python      : {sys.version.split()[0]}")
print(f"En Entorno Virtual     : {sys.prefix != sys.base_prefix}")
print("-" * 60)
print(f"Python 3 global (path) : {shutil.which('python3')}")
print(f"Conda                  : {check('conda')}")
print(f"Homebrew               : {check('brew')}")
print(f"Git                    : {check('git')}")
print(f"Xcode Select / Clang   : {check('clang')}")
print("-" * 60)

# Comprobar PATH para ver posibles residuos de Anaconda u otros gestores
path_env = os.environ.get("PATH", "")
print("Rutas detectadas en el PATH que podrían generar conflicto:")
conflicts = [p for p in path_env.split(":") if "anaconda" in p.lower() or "miniconda" in p.lower() or "homebrew" in p.lower()]
if conflicts:
    for c in conflicts:
        print(f"  - {c}")
else:
    print("  Ninguna ruta conflictiva detectada en PATH.")
print("=" * 60)