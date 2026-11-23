# Build a TimeNet Hackathon Entry

The example builds a room occupancy classifier for a facilities operator. At each evaluation point it receives the latest 16 sensor readings, including the current reading, and predicts whether the room is occupied at that point. The demo shows the sensor traces and the prediction so an operator can compare the result with the observed conditions.

The same workflow works for another task. Define the available input, the target, when both are measured, the decision the output supports and the metric before writing the connector.

## 1. Smoke-test TimeNet

The workspace contains two sibling checkouts. Keep training, evaluation and the demo in `timenet_hackathon/`. Use `TimeNet/` to develop the connector for the upstream dataset PR. The TimeNet checkout pins Python 3.13 for development. Install Git and a recent `uv` before running these commands from `timenet_hackathon/`.

```bash
cd ../TimeNet
make sync
export TIMENET_LOCAL_REGISTRY="$(mktemp -d)/registry"
uv run timenet-build build timenet/hello-world --no-isolation
uv run timenet info timenet/hello-world@1.0.0
```

The built-in dataset generates synthetic data without fetching a raw dataset. Environment setup still downloads dependencies. A successful smoke test creates a TimeF release and prints its metadata and counts. Keep using the same registry environment variable in shells that need to see that release. [TimeNet quickstart](https://docs.timenet.ai/get-started.html) · [Build CLI](https://docs.timenet.ai/cli/build.html)

## 2. Define the dataset and splits

The example uses **UCI Occupancy Detection** by Luis Candanedo, DOI `10.24432/C5X01N`, under **CC BY 4.0**. The source ships `datatraining.txt`, `datatest.txt` and `datatest2.txt`. Use the original download to preserve the file boundaries. Search the TimeNet catalog and open upstream PRs before naming or writing a connector. `uci/occupancy-detection` below is a proposed ID. [UCI dataset and license](https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection) · [TimeNet registry](https://registry.timenet.ai/) · [TimeNet pull requests](https://github.com/OpenTSLM/TimeNet/pulls)

| Item | Example configuration |
| --- | --- |
| Input channels | Temperature (°C), relative humidity (%), light (lux) and CO₂ (ppm). The source also provides a derived humidity ratio that this example omits. |
| Window | 16 consecutive readings from the same partition, oldest first. Start with a stride of 8 rows for training and 16 rows for validation and test. |
| Target | Map the `Occupancy` value of the last reading to `occupied` or `vacant`. Keep it in the task target. |
| Train and validation | Sort `datatraining.txt` by timestamp. Reserve the last 20% of rows for validation and construct windows separately within each partition. Check that both labels occur in the validation windows. |
| Test | Preserve both official test files. Build windows independently per file and report each file's results and a pooled score. |

Validate observed timestamp gaps and duplicate timestamps per file before windowing. Cut at discontinuities and preserve the source's time offsets. Use a regular axis only where the observed cadence is exact. Preserve irregular timestamps in TimeF and define how timing is represented in the model input. Document any discarded windows and missing values. Fit scaling parameters on training rows only and reuse them across splits. Keep source row IDs, filenames and timestamps for auditing without feeding them as predictors.

Check that no raw row or overlapping source time interval appears across partitions. Freeze the split and window rules before testing. In other projects use the true independence boundary, such as recording, machine or subject, or a chronological split for forecasting.

## 3. Build the connector and open the dataset PR

1. In the `TimeNet/` checkout, read `AGENTS.md`, `CONTRIBUTING.md` and the connector guide. Create a feature branch when you start implementation. For the example ID, use `packages/timenet-connectors/src/timenet_connectors/datasets/uci/occupancy_detection/` and add the `uci` package initializer if needed. Use `datasets/timenet/hello_world/` for the smallest in-tree reference covering classification and `datasets/chengsenwang/tsqa/` for fixture tests. [TimeNet connector guide](https://docs.timenet.ai/connectors.html)
2. Write `dataset.yaml` with the dataset ID, explicit version such as `1.0.0`, citation, source URL, license and a description of the windows and splits. In `connector.py`, implement `BaseConnector` for the UCI archive. Have `download(cache_dir)` fetch or reuse the archive, check its SHA-256 and return file references. Have `convert(raw_refs)` parse and validate locally. Export the class as `CONNECTOR` through `__init__.py`. Add connector-specific dependencies to `requirements.txt` if needed.
3. In `convert`, create a `TimeFDataset` using the card metadata. For each window, create one `Record` with a `Source` containing the four aligned signals. Give each signal a spec with its physical unit and an axis. Store the source filename, first and last timestamps and row identifiers as record metadata. Add the record before adding a `ClassificationTask` with that record in `inputs`, the terminal label in `targets`, a shared two-class `LabelSet` and the appropriate `Split`.
4. Test conversion against a tiny local fixture. Assert values and units survive conversion, the last row supplies the correct target, and no window crosses a file, split or gap boundary. Build and reload the real dataset. Check for expected tasks, channel order and class counts by split. Run the TimeNet checks below before opening the PR. [TimeF data model](https://docs.timenet.ai/data-model/index.html) · [TimeNet contributing guide](https://github.com/OpenTSLM/TimeNet/blob/main/CONTRIBUTING.md)

From `TimeNet/` after adding the connector:

```bash
uv run timenet-build build uci/occupancy-detection@1.0.0 --force
uv run timenet info uci/occupancy-detection@1.0.0
make check
make lint-fix
make test
make test-connectors
```

The isolated build installs declared connector dependencies. For rapid local iteration with those dependencies already installed, add `--no-isolation`. Keep generated TimeF artifacts and model checkpoints out of the connector PR. Push the feature branch to your GitHub fork and open a PR against `OpenTSLM/TimeNet`. Include the upstream dataset attribution, split and window rules, commands to rebuild and test results. Save the PR URL for the submission. [Build CLI](https://docs.timenet.ai/cli/build.html) · [Upstream TimeNet repository](https://github.com/OpenTSLM/TimeNet)

## 4. Adapt and fine-tune OpenTSLM

For Hill of Towie, follow the [operator audit and two-stage Flamingo plan](annotations/README.md#training-and-repository-decision). The steps below use the occupancy example.

Run TimeNet preparation and OpenTSLM training in separate `uv` environments. Use the local TimeNet environment to export one file per split, such as JSON Lines, then copy the exports to the model environment. Each row should include the stable source window key, signal names and units, arrays in a fixed channel order, timing information and the gold label. Preserve the source test filename for per-file scores. Keep labels and audit metadata separate from the model prompt.

This inspection snippet runs from `TimeNet/` with `uv run python` after the example dataset is built and the local registry variable is set:

```python
from timenet import TimeNet

dataset = TimeNet().load("uci/occupancy-detection@1.0.0")

for split, tasks in (
    ("train", dataset.get_train()),
    ("validation", dataset.get_validation()),
    ("test", dataset.get_test()),
):
    print(split, len(tasks))
    task = tasks[0]
    print([(signal.name, signal.to_numpy().shape)
           for signal in task.inputs[0].signals])
    print(task.targets[0].read())
```

1. **Check the model environment.** On a Linux NVIDIA GPU that supports the checkpoint's data type, set up the OpenTSLM repo and run its TSQA demo. Use `OpenTSLM/gemma-3-270m-tsqa-sp` as the initial checkpoint by setting `REPO_ID` in the demo. Check access to its underlying Gemma model before reserving GPU time. Record the OpenTSLM commit and the checkpoint revision. [OpenTSLM setup](https://github.com/OpenTSLM/OpenTSLM#building-and-finetuning-your-own-models) · [Checkpoint](https://huggingface.co/OpenTSLM/gemma-3-270m-tsqa-sp)

   ```bash
   git clone https://github.com/OpenTSLM/OpenTSLM.git
   cd OpenTSLM
   uv sync --all-groups
   uv run python demo/huggingface/01_test_hf_tsqa.py
   ```

2. **Write the dataset adapter.** Use the TSQA demo's batch contract. Create a `PromptWithAnswer` with a `TextPrompt` asking whether the room is occupied at the last reading, one `TextTimeSeriesPrompt` per channel in the same order for every sample, and a final `TextPrompt` asking for either `occupied` or `vacant`. Keep the numeric values in the time-series inputs. Mention the channel, original unit and cadence in prompt text, and retain physical levels with train-fitted scaling so absolute light and CO₂ levels remain distinguishable. Append the model EOS token to the supervised answer. Call `.to_dict()` on each example and use the demo's `extend_time_series_to_match_patch_size_and_aggregate` collator with `PATCH_SIZE`. [TSQA dataset adapter](https://github.com/OpenTSLM/OpenTSLM/blob/main/src/opentslm/time_series_datasets/TSQADataset.py) · [Prompt formatting](https://github.com/OpenTSLM/OpenTSLM/blob/main/src/opentslm/time_series_datasets/QADataset.py)
3. **Fine-tune the time-series components.** Load the checkpoint through `OpenTSLM.load_pretrained(..., device="cuda")`. For the Soft Prompt model, freeze the language backbone and train the encoder and projector with AdamW. Use batch size 1 for the first pass. In training mode, clear gradients, run `model.compute_loss(batch)`, backpropagate and step the optimizer. Start with a few batches, verify finite loss and gradients, and test that the saved weights reload. Then train for a short run, select a checkpoint on validation loss or macro F1, and save it with `model.store_to_file(path)`. Use the same base checkpoint with `model.load_from_file(path)` to restore. The upstream curriculum CLI's built-in stages use upstream datasets. Put the occupancy adapter and training loop in the project so the run uses the TimeF exports. [Soft Prompt model](https://github.com/OpenTSLM/OpenTSLM/blob/main/src/opentslm/model/llm/OpenTSLMSP.py) · [Upstream training loop](https://github.com/OpenTSLM/OpenTSLM/blob/main/curriculum_learning.py)

## 5. Evaluate and package the demo

- Train a logistic-regression baseline on summary features per window such as the latest value and mean of each channel. Score that baseline, the original checkpoint and the adapted checkpoint on the same held-out windows. Tune all choices on validation data, then freeze the configuration before the final test run.
- Decode with `model.generate(..., do_sample=False)` in evaluation mode with gradients disabled. Pass the prompt and signal data only. Use a fixed parser for the allowed labels. Count unparseable outputs as errors, report their rate separately and retain them in the evaluation denominator.
- Report macro F1, accuracy, occupied-class precision and recall, and a confusion matrix that accounts for invalid outputs. Include counts and class balance for each file. Keep the dataset version, code and model revisions, seed, preprocessing, training configuration and per-window predictions so another team member can reproduce the scores.
- Record a short demo with two held-out windows. Plot the signals with units and a shared time axis, show the prediction at the final timestamp, then reveal the label. Walk through one correct prediction and one error. State that results come from one office and explain which lighting or ventilation patterns might affect performance elsewhere.
- Publish the public showcase and complete the [submission checklist](README.md#submission-checklist).
