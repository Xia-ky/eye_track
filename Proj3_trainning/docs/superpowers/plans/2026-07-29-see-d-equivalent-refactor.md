# SEE-D Equivalent Clear Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a readable, JSON-driven SEE-D implementation with variable-depth inverted-residual backbones, Float32 training, Int8 QAT, dynamic checkpoint transfer, and a self-contained one-record AutoDL test.

**Architecture:** `Proj3_trainning/software_simplised` is the canonical implementation. It separates configuration, data, model, checkpoint, training, and reporting; the original `software` directories remain immutable behavior references. `Proj2_test_train/software_simplised` is generated from the canonical tree and adds only the full-record one-epoch AutoDL profile and lifecycle script.

**Tech Stack:** Python 3.8, PyTorch 1.8.2, CUDA 11.1, the repository's modified MinkowskiEngine, Tonic, NumPy, h5py, Bash, standard-library `unittest`.

**Repository note:** `E:\Project\zynq_cnn_proj` is not a Git worktree. Commit steps are replaced by source manifests, focused test runs, and SHA-256 checkpoints after every task.

---

## Locked file structure

Create or replace these hand-written files under
`Proj3_trainning/software_simplised`:

```text
train.py
evaluate.py
inspect_model.py
sync_proj2.py
esda/__init__.py
esda/config.py
esda/data/__init__.py
esda/data/dataset.py
esda/data/transforms.py
esda/data/voxel.py
esda/data/factory.py
esda/models/__init__.py
esda/models/sparse_ops.py
esda/models/blocks.py
esda/models/see_d.py
esda/models/quantization.py
esda/models/factory.py
esda/models/hawq/__init__.py
esda/models/hawq/quant_modules.py
esda/models/hawq/quant_utils.py
esda/engine/__init__.py
esda/engine/losses.py
esda/engine/metrics.py
esda/engine/trainer.py
esda/engine/checkpoint.py
esda/utils/__init__.py
esda/utils/logging.py
esda/utils/reproducibility.py
configs/see_d_float32.json
configs/see_d_int8_qat.json
configs/see_d_small_float32.json
tests/test_config.py
tests/test_model_plan.py
tests/test_checkpoint_mapping.py
tests/test_source_protection.py
tests/compare_with_original.py
README.md
requirements.txt
setup_env.sh
run.sh
```

Move the byte-preserved modified MinkowskiEngine build tree into:

```text
third_party/minkowski_engine/setup.py
third_party/minkowski_engine/MinkowskiEngine/
third_party/minkowski_engine/pybind/
third_party/minkowski_engine/src/
```

Generate these files under `Proj2_test_train/software_simplised`:

```text
run.sh
setup_env.sh
prepare_one_record.py
configs/see_d_one_record.json
README.md
SYNC_MANIFEST.json
```

The remaining canonical Python package and vendored backend are synchronized from
Proj3 rather than maintained independently.

### Task 1: Protect original code and establish a test harness

**Files:**
- Create: `Proj3_trainning/software_simplised/tests/test_source_protection.py`
- Create: `Proj3_trainning/software_simplised/tests/original_source_manifest.json`
- Create: `Proj3_trainning/software_simplised/tests/__init__.py`

- [ ] **Step 1: Write the manifest test**

Implement a standard-library test with this interface:

```python
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("original_source_manifest.json")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

class OriginalSourceProtectionTests(unittest.TestCase):
    def test_original_files_still_match_baseline(self):
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for relative_path, expected in entries.items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(expected, sha256(path), relative_path)
```

The manifest covers every file in `Proj3_trainning/software` and
`Proj2_test_train/software`, excluding generated `__pycache__`.

- [ ] **Step 2: Run the protection test**

Run:

```powershell
python -m unittest Proj3_trainning/software_simplised/tests/test_source_protection.py -v
```

Expected: `OK`.

- [ ] **Step 3: Record the starting simplified-tree inventory**

Write a second manifest entry group for the current `software_simplised` source so
later checkpoint-adapter work can trace every copied legacy file.

### Task 2: Implement strict JSON configuration with variable block count

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/__init__.py`
- Create: `Proj3_trainning/software_simplised/esda/config.py`
- Create: `Proj3_trainning/software_simplised/tests/test_config.py`
- Create: `Proj3_trainning/software_simplised/configs/see_d_float32.json`
- Create: `Proj3_trainning/software_simplised/configs/see_d_int8_qat.json`
- Create: `Proj3_trainning/software_simplised/configs/see_d_small_float32.json`

- [ ] **Step 1: Write failing configuration tests**

Cover these exact test methods: `test_paper_config_has_expected_explicit_blocks`,
`test_five_block_small_config_is_valid`, `test_empty_block_list_is_rejected`,
`test_adjacent_channel_mismatch_is_rejected`, `test_invalid_residual_is_rejected`,
`test_unknown_field_is_rejected`, `test_duplicate_name_is_rejected`, and
`test_duplicate_checkpoint_source_is_rejected`. The two valid cases assert exact
ordered block names; every invalid case uses `self.assertRaisesRegex(ConfigError,
expected_message)`.

The five-block configuration keeps source blocks `block_0`, `block_1`,
`block_3`, `block_5`, and `block_8`; this preserves channel transitions while
removing four same-width blocks.

- [ ] **Step 2: Confirm tests fail because the loader is absent**

Run:

```powershell
python -m unittest Proj3_trainning/software_simplised/tests/test_config.py -v
```

Expected: import failure for `esda.config`.

- [ ] **Step 3: Implement typed configuration and strict-key parsing**

Define these immutable dataclasses:

```python
@dataclass(frozen=True)
class BlockConfig:
    name: str
    checkpoint_source: Optional[str]
    in_channels: int
    expand_channels: int
    out_channels: int
    stride: int
    use_residual: bool

@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    quantization: QuantizationConfig
    runtime: RuntimeConfig
    output: OutputConfig
    source_path: Path

def load_config(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = parse_experiment_config(raw, source_path=path.resolve())
    validate_config(config)
    return config
```

Use `reject_unknown_keys(section, raw, allowed)` before constructing every
dataclass. Validate `N >= 1`, stride in `{1, 2}`, adjacent channels, residual
conditions, unique names, and unique non-null checkpoint sources.
`validate_config(config: ExperimentConfig) -> None` raises `ConfigError` with
the JSON path and offending values on the first structural error.

- [ ] **Step 4: Write the three complete configurations**

`see_d_float32.json` explicitly represents the original 9 blocks. The small
configuration contains the five source blocks listed above.
`see_d_int8_qat.json` uses the same model section and sets:

```json
"quantization": {
  "enabled": true,
  "shift_bit": 16,
  "bias_bit": 16,
  "conv1_bit": 8,
  "fix_bn_ratio": 0.3,
  "initialize_from": ""
}
```

- [ ] **Step 5: Run configuration tests**

Expected: all configuration tests pass without importing PyTorch, Tonic, or
MinkowskiEngine.

### Task 3: Add dependency-free model planning and inspection

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/models/__init__.py`
- Create: `Proj3_trainning/software_simplised/esda/models/plan.py`
- Create: `Proj3_trainning/software_simplised/tests/test_model_plan.py`
- Create: `Proj3_trainning/software_simplised/inspect_model.py`

- [ ] **Step 1: Write failing shape-plan tests**

Assert that the paper configuration produces 9 blocks, the small configuration
produces 5, and the final feature width is 64 in both. Assert exact channel and
stride values for every planned block.

- [ ] **Step 2: Implement the pure planning API**

```python
@dataclass(frozen=True)
class LayerPlan:
    name: str
    kind: str
    in_channels: int
    hidden_channels: Optional[int]
    out_channels: int
    stride: int
    use_residual: bool

def build_layer_plan(config: ExperimentConfig) -> List[LayerPlan]:
    layers = [stem_plan(config.model.stem)]
    layers.extend(block_plan(block) for block in config.model.blocks)
    layers.append(tail_plan(config.model.tail))
    return layers
```

This module has no numerical-framework imports and is therefore testable on the
local Windows environment. `format_layer_plan(plan: Sequence[LayerPlan]) -> str`
renders one fixed-width row per layer and a final block-count/cumulative-stride
summary.

- [ ] **Step 3: Implement `inspect_model.py`**

It loads JSON, prints the plan, total block count, cumulative spatial stride,
and—when PyTorch/MinkowskiEngine is available—actual parameter count.

- [ ] **Step 4: Run plan tests and both inspection commands**

Expected: `blocks=9` for the paper config and `blocks=5` for the small config.

### Task 4: Refactor the data contract without changing algorithms

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/data/__init__.py`
- Create: `Proj3_trainning/software_simplised/esda/data/dataset.py`
- Create: `Proj3_trainning/software_simplised/esda/data/transforms.py`
- Create: `Proj3_trainning/software_simplised/esda/data/voxel.py`
- Create: `Proj3_trainning/software_simplised/esda/data/factory.py`
- Create: `Proj3_trainning/software_simplised/tests/test_data_contract.py`

- [ ] **Step 1: Write characterization tests**

Use `unittest.skipUnless` when NumPy/Tonic/h5py are unavailable. On AutoDL,
compare original and refactored:

```python
self.assertTrue(np.array_equal(new_labels, old_labels))
self.assertEqual(new_voxels.dtype, np.float32)
np.testing.assert_array_equal(new_voxels, old_voxels)
self.assertEqual(new_voxels.shape, (30, 3, 60, 80))
```

- [ ] **Step 2: Implement the raw recording loader**

Copy the original H5 and label parsing behavior exactly, including structured
event dtype and polarity conversion. Resolve train and validation records under
`data_root/train/<record_id>`.

- [ ] **Step 3: Implement transforms with explicit contracts**

Retain the original formulas for `ScaleLabel`, `TemporalSubsample`,
`NormalizeLabel`, `SliceByTimeEventsTargets`, `SliceLongEventsToShort`,
`EventSlicesToVoxelGrid`, `Flip`, and `Shift`. Document time units and tensor
axes at each boundary.

- [ ] **Step 4: Implement one-cache-layer dataset construction**

`build_datasets(config)` creates `SlicedDataset`, wraps it exactly once in
`DiskCachedDataset`, and applies augmentation outside the cache only for the
training split.

- [ ] **Step 5: Run local compile/tests and AutoDL characterization test**

Local expected result: dependency-free tests pass and dependency-requiring
tests skip. AutoDL expected result: no skips and exact array equality.

### Task 5: Implement the dynamic Float32 SEE-D model

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/models/sparse_ops.py`
- Create: `Proj3_trainning/software_simplised/esda/models/blocks.py`
- Create: `Proj3_trainning/software_simplised/esda/models/see_d.py`
- Create: `Proj3_trainning/software_simplised/esda/models/factory.py`
- Create: `Proj3_trainning/software_simplised/tests/test_model_structure.py`

- [ ] **Step 1: Write AutoDL structure tests**

For both 9-block and 5-block configs, assert block order, residual flags,
parameter dimensions, output shape `[2, 30, 2]`, and successful backward pass.

- [ ] **Step 2: Copy `dense_to_sparse` exactly**

Keep the original coordinate selection and feature extraction:

```python
non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
select_indices = non_zero_indices.split(1, dim=1)
features = torch.squeeze(dense[select_indices], dim=-2)
return non_zero_indices, features
```

- [ ] **Step 3: Implement explicit sparse blocks**

`InvertedResidualBlock` takes `expand_channels` directly and builds the same
1x1 convolution, 3x3 channelwise convolution, 1x1 projection, BN, ReLU6, and
optional residual as the original.

- [ ] **Step 4: Implement dynamic `SeeDModel`**

Use a `ModuleDict` keyed by JSON block name plus a separate ordered name tuple.
Forward is:

```python
x = x.reshape(batch_size * time_steps, channels, height, width)
x = x.permute(0, 2, 3, 1)
coordinates, features = dense_to_sparse(x)
x = SparseTensor(features=features.contiguous(),
                 coordinates=coordinates.int().contiguous(),
                 device=self.device)
x = self.stem(x)
for name in self.block_order:
    x = self.blocks[name](x)
x = self.tail(x)
x = self.pool(x).F.reshape(batch_size, time_steps, -1)
x, _ = self.gru(x)
return self.head(x)
```

- [ ] **Step 5: Run structure tests on AutoDL**

Expected: both depths pass forward/backward; paper model parameter count is
177970.

### Task 6: Implement dynamic checkpoint discovery and transfer

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/engine/__init__.py`
- Create: `Proj3_trainning/software_simplised/esda/engine/checkpoint.py`
- Create: `Proj3_trainning/software_simplised/tests/test_checkpoint_mapping.py`

- [ ] **Step 1: Write dependency-free key-mapping tests**

Test legacy Float32 keys `features.<index>.*`, legacy quantized keys
`model.blocks.<index>.*`, and new keys `blocks.<name>.*`. Remove a middle
target block and prove the following target block still maps from its explicit
`checkpoint_source`, not its new list index.

- [ ] **Step 2: Implement discovery and planning**

```python
@dataclass(frozen=True)
class CheckpointPlan:
    key_map: Mapping[str, str]
    loaded_blocks: Tuple[str, ...]
    removed_source_blocks: Tuple[str, ...]
    initialized_blocks: Tuple[str, ...]
    errors: Tuple[str, ...]

```

Implement `discover_source_blocks(keys: Iterable[str]) -> Mapping[str, str]`
with anchored regular expressions for the three supported key layouts.
Implement `plan_checkpoint_transfer(source_keys, target_keys, blocks) ->
CheckpointPlan` by substituting only the block prefix associated with each
explicit `checkpoint_source`; non-block keys map by their declared legacy
prefix table.

- [ ] **Step 3: Implement strict resume and explicit initialize**

`resume_checkpoint` requires exact keys and shapes. `initialize_checkpoint`
applies the plan, rejects every shape mismatch, loads null-source blocks only
from model initialization, and writes `checkpoint_load_report.json`.

- [ ] **Step 4: Save self-describing checkpoints**

Save `model_state`, `optimizer_state`, `epoch`, `metrics`,
`resolved_config`, and `block_manifest`. Never save a bare state dict for new
runs.

- [ ] **Step 5: Run mapping tests locally and tensor-loading tests on AutoDL**

Expected: the 5-block model inherits compatible blocks 0, 1, 3, 5, and 8 from
the legacy baseline checkpoint with an explicit report.

### Task 7: Isolate and adapt Int8 QAT

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/models/hawq/__init__.py`
- Copy unchanged: `Proj3_trainning/software_simplised/model/HAWQ_quant_module/quant_modules.py`
  to `Proj3_trainning/software_simplised/esda/models/hawq/quant_modules.py`
- Copy unchanged: `Proj3_trainning/software_simplised/model/HAWQ_quant_module/quant_utils.py`
  to `Proj3_trainning/software_simplised/esda/models/hawq/quant_utils.py`
- Create: `Proj3_trainning/software_simplised/esda/models/quantization.py`
- Create: `Proj3_trainning/software_simplised/tests/test_quantized_structure.py`

- [ ] **Step 1: Verify copied HAWQ files are byte-identical**

Record both source and destination SHA-256 values in the test output.

- [ ] **Step 2: Implement dynamic quantized blocks**

Build one quantized bottleneck per current JSON block by reading the named
Float32 block from `SeeDModel.blocks`. Do not rebuild blocks from a grouped
backbone count.

- [ ] **Step 3: Preserve the original quantization boundaries**

Quantize input, stem, each sparse bottleneck, tail, and global average pool.
Reuse the Float32 GRU and linear head exactly as the original
`HAWQ_mobilenetv2.py`.

- [ ] **Step 4: Run AutoDL tests**

Test 9-block and 5-block construction, one forward/backward pass, freeze and
unfreeze, strict resume, and official-checkpoint initialization report.

### Task 8: Refactor losses, metrics, training, and reporting

**Files:**
- Create: `Proj3_trainning/software_simplised/esda/engine/losses.py`
- Create: `Proj3_trainning/software_simplised/esda/engine/metrics.py`
- Create: `Proj3_trainning/software_simplised/esda/engine/trainer.py`
- Create: `Proj3_trainning/software_simplised/esda/utils/__init__.py`
- Create: `Proj3_trainning/software_simplised/esda/utils/reproducibility.py`
- Create: `Proj3_trainning/software_simplised/esda/utils/logging.py`
- Create: `Proj3_trainning/software_simplised/tests/test_metrics.py`

- [ ] **Step 1: Write numerical metric tests**

Use small fixed tensors to prove weighted MSE uses weights `[4/3, 1]`, P5/P10
use only the last time step, and distance uses all time steps scaled by
`80 x 60`.

- [ ] **Step 2: Implement pure metric functions**

Expose:

```python
def weighted_mse(prediction, target, weights):
    return (((prediction - target) ** 2) * weights).mean()

def pixel_accuracy(prediction, target, width, height, tolerances):
    delta = (target[:, -1, :2] - prediction[:, -1, :2]).clone()
    delta[:, 0] *= width
    delta[:, 1] *= height
    distance = torch.norm(delta, dim=-1)
    return {p: (distance < p).float().mean().item() for p in tolerances}

def mean_pixel_distance(prediction, target, width, height):
    delta = (target[..., :2] - prediction[..., :2]).clone()
    delta[..., 0] *= width
    delta[..., 1] *= height
    return torch.norm(delta, dim=-1).mean().item()
```

- [ ] **Step 3: Implement train and validation epochs**

Preserve Adam, all-step loss, last-step P metrics, all-step distance, QAT
freeze/unfreeze timing, and progress text. Return typed dictionaries instead of
nested mutable recorder state.

- [ ] **Step 4: Implement durable outputs**

Write `metrics.csv`, `report.json`, resolved configuration, environment
summary, best-P10 checkpoint, best-loss checkpoint, and last checkpoint using
atomic replacement.

- [ ] **Step 5: Run metric tests locally and epoch tests on AutoDL**

Expected: exact fixed-tensor values and successful one-batch optimization.

### Task 9: Implement training, evaluation, and official comparison entry points

**Files:**
- Create: `Proj3_trainning/software_simplised/train.py`
- Create: `Proj3_trainning/software_simplised/evaluate.py`
- Create: `Proj3_trainning/software_simplised/tests/compare_with_original.py`

- [ ] **Step 1: Implement minimal CLIs**

Both entry points accept only `--config`; runtime overrides are made by copying
and editing JSON so every run remains self-describing.

- [ ] **Step 2: Implement the orchestration order**

Load/validate config, seed, build data, build Float32 or QAT model, apply
checkpoint policy, create Adam, train/evaluate, and write final report.

- [ ] **Step 3: Implement stage-by-stage comparison hooks**

Register forward hooks for stem, every named block, tail, pool, GRU, and head.
Compare original and new default models with `torch.testing.assert_close`, and
write a JSON report containing maximum absolute and relative error per stage.

- [ ] **Step 4: Run default equivalence on AutoDL**

Expected: data equality, matching parameter count, compatible checkpoint
mapping, forward tensors within stated tolerance, and matching metrics.

### Task 10: Isolate the modified MinkowskiEngine and environment setup

**Files:**
- Move without content changes: `setup.py`, `MinkowskiEngine/`, `pybind/`, and
  `src/` into `Proj3_trainning/software_simplised/third_party/minkowski_engine/`
- Create: `Proj3_trainning/software_simplised/requirements.txt`
- Create: `Proj3_trainning/software_simplised/setup_env.sh`

- [ ] **Step 1: Hash the vendored backend before moving**

Generate a manifest of all four source paths.

- [ ] **Step 2: Move the build tree and verify byte identity**

Every moved file must match its pre-move SHA-256.

- [ ] **Step 3: Update environment setup**

Keep Python 3.8, PyTorch 1.8.2+cu111, CUDA toolkit 11.1,
setuptools 59.5.0, NumPy 1.21.0, OpenBLAS, Ninja, and
`TORCH_CUDA_ARCH_LIST=8.6`. Compile from
`third_party/minkowski_engine/setup.py --force_cuda`.

- [ ] **Step 4: Run shell static checks**

Run `bash -n setup_env.sh` and `bash -n run.sh` on AutoDL or WSL.

### Task 11: Build the self-contained Proj2 full-record test

**Files:**
- Create: `Proj3_trainning/software_simplised/sync_proj2.py`
- Create: `Proj2_test_train/software_simplised/prepare_one_record.py`
- Create: `Proj2_test_train/software_simplised/configs/see_d_one_record.json`
- Create: `Proj2_test_train/software_simplised/run.sh`
- Create: `Proj2_test_train/software_simplised/setup_env.sh`
- Create: `Proj2_test_train/software_simplised/README.md`
- Generate: `Proj2_test_train/software_simplised/SYNC_MANIFEST.json`

- [ ] **Step 1: Implement synchronization**

Copy the canonical package, entry points, requirements, and vendored backend;
exclude caches, tests not needed at runtime, outputs, and Proj3-only configs.
Write relative path, byte count, and SHA-256 for every copied file.

- [ ] **Step 2: Implement full-record list preparation**

Validate complete `1_2.h5`, `1_2/label.txt`, `1_6.h5`, and `1_6/label.txt`,
then write one-line train and validation lists. Do not copy or crop H5 data.

- [ ] **Step 3: Implement safe lifecycle script**

Run one Float32 epoch, save all output and console text, require a checkpoint
and passed `report.json`, persist results atomically, and request shutdown only
when `AUTO_SHUTDOWN=1` and training succeeded. Failed runs remain online.

- [ ] **Step 4: Verify synchronization**

Run:

```powershell
python Proj3_trainning/software_simplised/sync_proj2.py --check
```

Expected: every manifest entry matches and no unexpected core file exists.

### Task 12: Documentation, cleanup, and final verification

**Files:**
- Replace: `Proj3_trainning/software_simplised/README.md`
- Create: `Proj3_trainning/software_simplised/FILE_GUIDE.md`
- Update: `Proj3_trainning/README.md`

- [ ] **Step 1: Write the shortest modification path**

Document copying `see_d_float32.json`, deleting same-width residual blocks,
maintaining channel continuity, inspecting the model, running Proj2, then
starting Proj3.

- [ ] **Step 2: Write the detailed reading path**

Trace H5 events through voxelization, sparse conversion, every configured
block, pooling, GRU, loss, metrics, and checkpoints. Explain every hand-written
file.

- [ ] **Step 3: Remove superseded simplified legacy files**

Only after all new tests pass, delete old `main.py`, `int_inference.py`,
`test.py`, `dataset/`, `model/`, `utils/`, and old split configs from
`software_simplised`. Never delete from either original `software`.

- [ ] **Step 4: Run local verification**

```powershell
python -m unittest discover -s Proj3_trainning/software_simplised/tests -v
python -m compileall -q Proj3_trainning/software_simplised
python Proj3_trainning/software_simplised/inspect_model.py --config Proj3_trainning/software_simplised/configs/see_d_float32.json
python Proj3_trainning/software_simplised/inspect_model.py --config Proj3_trainning/software_simplised/configs/see_d_small_float32.json
python Proj3_trainning/software_simplised/sync_proj2.py --check
```

Expected: standard-library tests pass, unavailable numerical dependencies are
reported as skips, compileall passes, inspection reports 9 and 5 blocks, and
Proj2 hashes match.

- [ ] **Step 5: Run AutoDL gate**

Run Proj2 with `AUTO_SHUTDOWN=0`, inspect its report/checkpoint, run original
versus refactored equivalence, then test `AUTO_SHUTDOWN=1`. Only after all gates
pass should the Proj3 full-data configuration be started.
