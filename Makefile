ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PATH := $(HOME)/.local/bin:$(PATH)
export MUJOCO_GL := osmesa
export PYOPENGL_PLATFORM := osmesa
export PYTHONPATH := $(ROOT)
export LD_LIBRARY_PATH := $(ROOT)/.deps/usr/lib/x86_64-linux-gnu:$(ROOT)/.venv/lib/python3.12/site-packages/mujoco:$(LD_LIBRARY_PATH)
PY := $(ROOT)/.venv/bin/python

.PHONY: setup demo serve verify video

setup:
	bash $(ROOT)/scripts/setup.sh

demo:
	$(PY) -m washsim demo

serve:
	$(PY) -m dashboard.server

verify:
	$(PY) -m pytest
	$(PY) -m washsim verify

video:
	$(PY) -m washsim video
