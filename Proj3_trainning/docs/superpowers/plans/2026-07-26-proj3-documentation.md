# Proj3_trainning Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete file guide and teaching-oriented comments without changing training behavior.

**Architecture:** Keep the project-level explanation in root Markdown files. Add comments only to AutoDL wrapper scripts and the few local adaptation points in the upstream training entry, preserving upstream source compatibility.

**Tech Stack:** Markdown, Bash, Python/PyTorch/MLflow.

---

### Task 1: Add project documentation

**Files:**
- Create: `Proj3_trainning/FILE_GUIDE.md`
- Modify: `Proj3_trainning/README.md`

- [x] Document the directory structure, execution flow, outputs, and responsibility of every file or homogeneous file group.
- [x] Link the file guide from the main README and add a recommended reading order.

### Task 2: Comment the AutoDL wrappers

**Files:**
- Modify: `Proj3_trainning/run.sh`
- Modify: `Proj3_trainning/setup_env.sh`
- Modify: `Proj3_trainning/requirements-esda-software.txt`

- [x] Explain strict mode, environment overrides, preflight checks, atomic persistence, exit trapping, training exit-code capture, checkpoint verification, CUDA architecture selection, and pinned dependencies.
- [x] Preserve every executable command and default value.

### Task 3: Explain local ESDA adaptations

**Files:**
- Modify: `Proj3_trainning/software/main.py`

- [x] Add comments around configurable dataset/cache roots, MLflow experiment creation, local artifact resolution, optional augmentation, and custom CLI arguments.
- [x] Do not alter executable Python statements.

### Task 4: Static verification

**Files:**
- Verify all modified files.

- [x] Attempt Bash syntax checks; local `bash` is an unconfigured WSL launcher, then verify LF/no-BOM and unchanged Shell executable statements instead.
- [x] Run Python compile/AST checking for `software/main.py`.
- [x] Confirm documentation references and inspect the final changes.
