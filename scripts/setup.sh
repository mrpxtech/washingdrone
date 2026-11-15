#!/usr/bin/env bash
# Prepare a project-local virtualenv. System packages are optional; this script
# falls back to a user-space pip bootstrap and a local OSMesa library.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$HOME/.local/bin"

if ! command -v make >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/make" ]]; then
  mkdir -p "$HOME/.cache/debs"
  (cd "$HOME/.cache/debs" && apt-get download make)
  rm -rf /tmp/make-extract
  mkdir -p /tmp/make-extract
  dpkg-deb -x "$HOME/.cache/debs"/make_*.deb /tmp/make-extract
  cp /tmp/make-extract/usr/bin/make "$HOME/.local/bin/make"
  chmod +x "$HOME/.local/bin/make"
fi

# ctypes.util.find_library("OSMesa") only consults ldconfig. A user-local
# library is invisible there, so advertise the extracted copy.
if [[ ! -x "$HOME/.local/bin/ldconfig" ]]; then
  cat > "$HOME/.local/bin/ldconfig" << EOF
#!/bin/sh
if [ "\$1" = "-p" ]; then
  /sbin/ldconfig -p
  echo "	libOSMesa.so.8 (libc6,x86-64) => ${ROOT}/.deps/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"
  echo "	libOSMesa.so (libc6,x86-64) => ${ROOT}/.deps/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"
else
  exec /sbin/ldconfig "\$@"
fi
EOF
  chmod +x "$HOME/.local/bin/ldconfig"
fi

if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv .venv; then
    python3 -m venv --without-pip .venv
  fi
fi
if [[ ! -x .venv/bin/pip ]]; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
fi
.venv/bin/python -m pip install --upgrade pip
# The published mujoco wheel SIGILLs on this VM's CPU (no AVX). Keep a working
# source build and only install the wheel when import fails.
grep -v -E '^mujoco([=<>]|$)' requirements.txt | .venv/bin/python -m pip install -r /dev/stdin
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="${ROOT}/.deps/usr/lib/x86_64-linux-gnu:${ROOT}/.venv/lib/python3.12/site-packages/mujoco${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! .venv/bin/python -c 'import mujoco'; then
  .venv/bin/python -m pip install 'mujoco==3.12.0' || true
fi
if ! .venv/bin/python -c 'import mujoco'; then
  echo "MuJoCo did not import. On a CPU without AVX the PyPI wheel raises SIGILL."
  echo "Build MuJoCo 3.12.0 from source with -march=x86-64 -mno-avx -mno-avx2."
  echo "See README.md, section Installation."
  exit 1
fi

if [[ ! -e .deps/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 ]] && ! /sbin/ldconfig -p 2>/dev/null | grep -q 'libOSMesa.so'; then
  mkdir -p "$HOME/.cache/debs" .deps
  (cd "$HOME/.cache/debs" && apt-get download libosmesa6 libllvm20)
  dpkg-deb -x "$HOME/.cache/debs"/libosmesa6_*.deb .deps
  dpkg-deb -x "$HOME/.cache/debs"/libllvm20_*.deb .deps
  ln -sfn libOSMesa.so.8 .deps/usr/lib/x86_64-linux-gnu/libOSMesa.so
fi

echo "Python environment is ready: $ROOT/.venv"
echo "MuJoCo: $(.venv/bin/python -c 'import mujoco; print(mujoco.__version__)')"
echo "If rendering still cannot find OSMesa, run:"
echo "  sudo apt-get install -y python3.12-venv make ffmpeg libosmesa6 libllvm20"
