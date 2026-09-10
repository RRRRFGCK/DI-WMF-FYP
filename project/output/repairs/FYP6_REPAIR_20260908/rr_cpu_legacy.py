"""Torch-free copies of the unchanged legacy NumPy/SciPy RR operations.

An AST equality check verifies every copied function body against the original
source. The only substitution is PublishedCorrEncoder1D.input_length -> 288,
independently checked against the actual model class definition. Importing the
training runner just for these functions needlessly loads CUDA DLLs on Windows.
"""
from pathlib import Path
import ast
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, hilbert, sosfiltfilt


def _finite_signal(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Signal contains no finite samples")
    if not finite.all():
        positions = np.arange(values.size)
        values[~finite] = np.interp(positions[~finite], positions[finite], values[finite])
    return values


def _normalise_pair(ppg, respiration) -> tuple[np.ndarray, np.ndarray]:
    ppg = _finite_signal(ppg)
    respiration = _finite_signal(respiration)
    ppg = (ppg - ppg.mean()) / max(ppg.std(), 1e-8)
    low = float(respiration.min())
    high = float(respiration.max())
    respiration = (respiration - low) / max(high - low, 1e-8)
    return ppg.astype(np.float32), respiration.astype(np.float32)


def load_bidmc_subjects(path: Path, target_hz: int = 30) -> list[dict]:
    contents = loadmat(path, simplify_cells=True)
    subjects = []
    for subject_index, record in enumerate(contents["data"]):
        ppg_record = record["ppg"]
        respiration_record = record["ref"]["resp_sig"]["imp"]
        ppg_hz = int(round(float(ppg_record["fs"])))
        respiration_hz = int(round(float(respiration_record["fs"])))
        ppg = resample_poly(_finite_signal(ppg_record["v"]), target_hz, ppg_hz)
        respiration = resample_poly(
            _finite_signal(respiration_record["v"]), target_hz, respiration_hz
        )
        usable = min(ppg.size, respiration.size)
        usable = min(usable, 50 * 288)
        ppg, respiration = _normalise_pair(ppg[:usable], respiration[:usable])
        subjects.append(
            {
                "subject": subject_index,
                "ppg": ppg,
                "respiration": respiration,
                "sampling_hz": target_hz,
            }
        )
    return subjects


def respiratory_rate(signal: np.ndarray, sampling_hz: int = 30) -> float:
    centred = signal - signal.mean()
    frequencies = np.fft.rfftfreq(centred.size, d=1.0 / sampling_hz)
    power = np.abs(np.fft.rfft(centred)) ** 2
    respiratory_band = (frequencies >= 0.08) & (frequencies <= 0.8)
    if not respiratory_band.any():
        return float("nan")
    band_indices = np.flatnonzero(respiratory_band)
    peak = band_indices[int(np.argmax(power[respiratory_band]))]
    return float(60.0 * frequencies[peak])


def bandpass(values: np.ndarray, sampling_hz: int, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sampling_hz, output="sos")
    return sosfiltfilt(sos, values).astype(np.float64)


def respiratory_envelope(ppg: np.ndarray, sampling_hz: int) -> np.ndarray:
    cardiac = bandpass(ppg, sampling_hz, 0.8, min(4.0, 0.45 * sampling_hz))
    envelope = np.abs(hilbert(cardiac))
    return bandpass(envelope, sampling_hz, 0.08, 0.8)


def verify_legacy_cpu_equivalence(root: Path):
    class ReplaceModelLength(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id=="PublishedCorrEncoder1D" and node.attr=="input_length":
                return ast.Constant(value=288)
            return self.generic_visit(node)
    local_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    local = {node.name:node for node in local_tree.body if isinstance(node,ast.FunctionDef)}
    checks = {}
    sources = {
        "run_correncoder_regression.py":("_finite_signal","_normalise_pair","load_bidmc_subjects","respiratory_rate"),
        "evaluate_classical_rr_baselines.py":("bandpass","respiratory_envelope"),
    }
    for filename,names in sources.items():
        tree=ast.parse((root/filename).read_text(encoding="utf-8-sig"))
        original={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        for name in names:
            expected=ReplaceModelLength().visit(original[name])
            first=ast.dump(ast.Module(body=expected.body,type_ignores=[]),include_attributes=False)
            second=ast.dump(ast.Module(body=local[name].body,type_ignores=[]),include_attributes=False)
            if first!=second:
                raise RuntimeError(f"Copied CPU body no longer exactly matches original: {filename}:{name}")
            checks[name]="AST_body_identical"
    model_tree=ast.parse((root/"domain_mf"/"models.py").read_text(encoding="utf-8-sig"))
    model=next(node for node in model_tree.body if isinstance(node,ast.ClassDef) and node.name=="PublishedCorrEncoder1D")
    length=next(node.value.value for node in model.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=="input_length" for target in node.targets))
    assert length==288
    checks["input_length"]="288_verified_from_actual_model_AST"
    return checks
