from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent import sandbox

_LABEL_COLUMN_CANDIDATES = ("target", "label", "class", "y")


@dataclass
class DatasetDescription:
    columns: list
    row_count: int
    label_column: str
    label_column_is_explicit: bool  # True when literally named target/label/class/y

    def as_prompt_note(self) -> str:
        preview = ", ".join(self.columns[:12])
        if len(self.columns) > 12:
            preview += ", ..."
        if self.label_column_is_explicit:
            guidance = f"The label column is '{self.label_column}'."
        else:
            guidance = (
                f"No column is named 'target'/'label'/'class', so the LAST column "
                f"('{self.label_column}') is the label. Select it by position "
                f"(e.g. `y = df.iloc[:, -1]`, `X = df.iloc[:, :-1]`) rather than by name."
            )
        return (
            f"The dataset CSV has {self.row_count} data row(s) and {len(self.columns)} "
            f"column(s): {preview}. {guidance}"
        )


@dataclass
class CustomInputs:
    algorithm: str
    display_name: str
    dataset_name: str
    baseline_code: str
    dataset_csv: str
    description: DatasetDescription
    baseline_accuracy: Optional[float]
    baseline_warning: Optional[str]


def sanitize_algorithm_name(raw: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", raw.strip()).strip("_").lower()
    return cleaned or "custom_algorithm"


def load_code_file(path: str) -> tuple[str, str]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise RuntimeError(f"no such file: {file_path}")
    if file_path.suffix.lower() not in (".py", ".txt"):
        raise RuntimeError(f"expected a Python file (.py), got '{file_path.suffix}'")

    try:
        code = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"failed to read {file_path}: {e}") from e

    if not code.strip():
        raise RuntimeError(f"{file_path} is empty")

    return sanitize_algorithm_name(file_path.stem), code


def load_dataset_file(path: str) -> str:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise RuntimeError(f"no such file: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise RuntimeError(f"expected a .csv file, got '{file_path.suffix}'")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # A CSV exported from Excel is often cp1252 rather than UTF-8.
        try:
            content = file_path.read_text(encoding="latin-1")
        except OSError as e:
            raise RuntimeError(f"failed to read {file_path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"failed to read {file_path}: {e}") from e

    if not content.strip():
        raise RuntimeError(f"{file_path} is empty")
    return content


def describe_dataset(csv_text: str) -> DatasetDescription:
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        raise RuntimeError("the CSV has no header row") from None

    columns = [c.strip() for c in header]
    if not columns or all(not c for c in columns):
        raise RuntimeError("the CSV's header row is empty")

    row_count = 0
    for row in reader:
        if not row:
            continue  # tolerate blank trailing lines
        if len(row) != len(columns):
            raise RuntimeError(
                f"malformed CSV at data row {row_count + 1}: expected {len(columns)} "
                f"column(s), got {len(row)}"
            )
        row_count += 1

    if row_count == 0:
        raise RuntimeError("the CSV has a header row but no data rows")
    if len(columns) < 2:
        raise RuntimeError(
            "the CSV needs at least two columns (one or more features plus a label column)"
        )

    lowered = [c.lower() for c in columns]
    label_column = columns[-1]
    explicit = False
    for candidate in _LABEL_COLUMN_CANDIDATES:
        if candidate in lowered:
            label_column = columns[lowered.index(candidate)]
            explicit = True
            break

    return DatasetDescription(
        columns=columns,
        row_count=row_count,
        label_column=label_column,
        label_column_is_explicit=explicit,
    )


async def validate_baseline(code: str, dataset_csv: str) -> tuple[Optional[float], Optional[str]]:
    if "def run" not in code:
        raise RuntimeError(
            "the baseline file must define an entry point:\n\n"
            "    def run(data_path: str) -> dict:\n"
            "        # load the CSV at data_path, split it into train/test yourself,\n"
            "        # train, evaluate, and return:\n"
            "        #   {'accuracy': <0-1>, 'loss': <float>,\n"
            "        #    'training_time': <seconds in fit>, 'inference_time': <seconds in predict>}\n\n"
            "Returning a bare accuracy float also works, but then training and inference "
            "time can't be measured separately."
        )

    result = await sandbox.run(code, dataset_csv, sandbox.default_time_limit())

    if result.primary.syntax_errors:
        raise RuntimeError(f"the baseline code has a syntax error:\n{result.stderr}")
    if result.primary.runtime_errors:
        raise RuntimeError(
            f"the baseline code failed to run against this dataset:\n{result.stderr}\n\n"
            "Check that it reads the CSV at the path it's given and returns accuracy."
        )

    accuracy = result.primary.model_accuracy
    warning = None
    if accuracy <= 0.0:
        warning = (
            "the baseline ran but scored an accuracy of 0.0 — check it's returning real "
            "test-set accuracy in the range 0-1"
        )
    elif accuracy >= 0.999:
        warning = (
            f"the baseline already scores {accuracy:.4f}, leaving almost no headroom for the "
            "agents to demonstrate an improvement in accuracy"
        )
    return accuracy, warning


async def prepare_custom_inputs(
    code_path: str, data_path: str, display_name: Optional[str] = None
) -> CustomInputs:
    algorithm, code = load_code_file(code_path)
    dataset_csv = load_dataset_file(data_path)
    description = describe_dataset(dataset_csv)

    print(f"  dataset: {description.row_count} row(s), {len(description.columns)} column(s); "
          f"label column '{description.label_column}'"
          f"{'' if description.label_column_is_explicit else ' (inferred: last column)'}")
    print("  validating the baseline against the dataset in the sandbox...")

    accuracy, warning = await validate_baseline(code, dataset_csv)
    print(f"  baseline runs: accuracy={accuracy:.4f}")
    if warning:
        print(f"  WARNING: {warning}")

    return CustomInputs(
        algorithm=algorithm,
        display_name=display_name or Path(code_path).stem,
        dataset_name=Path(data_path).name,
        baseline_code=code,
        dataset_csv=dataset_csv,
        description=description,
        baseline_accuracy=accuracy,
        baseline_warning=warning,
    )
