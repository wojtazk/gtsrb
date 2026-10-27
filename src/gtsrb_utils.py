"""Wspólne funkcje i stałe projektu GTSRB (używane przez wszystkie notatniki).

Import w notatniku (notatniki leżą w katalogu notebooks/):

    import sys; sys.path.append("../src")
    import gtsrb_utils as gu

Dane: rozpakowany zbiór GTSRB (wersja z Kaggle / Zenodo) w katalogu data/GTSRB:

    data/GTSRB/
    ├── Train/0 … Train/42/   obrazy treningowe, nazwa pliku: <klasa>_<seria>_<klatka>.png
    ├── Test/                 obrazy testowe
    ├── Meta/                 wzorcowy obraz każdej klasy
    ├── Train.csv  Test.csv  Meta.csv

Inną lokalizację można wskazać zmienną środowiskową GTSRB_DIR.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Ścieżki projektu (liczone względem tego pliku, więc działają z każdego miejsca)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GTSRB_DIR = Path(os.environ.get("GTSRB_DIR", DATA_DIR / "GTSRB"))
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42

TRAIN_CSV_COLUMNS = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
META_CSV_COLUMNS = ["Path", "ClassId", "ShapeId", "ColorId", "SignId"]

# --------------------------------------------------------------------------
# Klasy GTSRB: numer klasy -> polska nazwa
# --------------------------------------------------------------------------
CLASS_NAMES_PL = [
    "ograniczenie 20 km/h",                     # 0
    "ograniczenie 30 km/h",                     # 1
    "ograniczenie 50 km/h",                     # 2
    "ograniczenie 60 km/h",                     # 3
    "ograniczenie 70 km/h",                     # 4
    "ograniczenie 80 km/h",                     # 5
    "koniec ograniczenia 80 km/h",              # 6
    "ograniczenie 100 km/h",                    # 7
    "ograniczenie 120 km/h",                    # 8
    "zakaz wyprzedzania",                       # 9
    "zakaz wyprzedzania ciężarówkom",           # 10
    "pierwszeństwo na skrzyżowaniu",            # 11
    "droga z pierwszeństwem",                   # 12
    "ustąp pierwszeństwa",                      # 13
    "STOP",                                     # 14
    "zakaz ruchu w obu kierunkach",             # 15
    "zakaz wjazdu ciężarówek",                  # 16
    "zakaz wjazdu",                             # 17
    "inne niebezpieczeństwo",                   # 18
    "niebezpieczny zakręt w lewo",              # 19
    "niebezpieczny zakręt w prawo",             # 20
    "niebezpieczne zakręty",                    # 21
    "nierówna droga",                           # 22
    "śliska jezdnia",                           # 23
    "zwężenie jezdni z prawej",                 # 24
    "roboty drogowe",                           # 25
    "sygnalizacja świetlna",                    # 26
    "przejście dla pieszych",                   # 27
    "dzieci",                                   # 28
    "rowerzyści",                               # 29
    "oszronienie / śnieg",                      # 30
    "dzikie zwierzęta",                         # 31
    "koniec zakazów",                           # 32
    "nakaz skrętu w prawo",                     # 33
    "nakaz skrętu w lewo",                      # 34
    "nakaz jazdy prosto",                       # 35
    "prosto lub w prawo",                       # 36
    "prosto lub w lewo",                        # 37
    "nakaz jazdy z prawej strony",              # 38
    "nakaz jazdy z lewej strony",               # 39
    "ruch okrężny",                             # 40
    "koniec zakazu wyprzedzania",               # 41
    "koniec zakazu wyprzedzania ciężarówkom",   # 42
]

GROUP_ORDER = [
    "ograniczenia prędkości",
    "zakazy",
    "odwołania",
    "pierwszeństwo",
    "ostrzegawcze",
    "nakazy",
]

_GROUP_RANGES = {
    "ograniczenia prędkości": list(range(0, 9)),
    "zakazy": [9, 10, 15, 16, 17],
    "odwołania": [32, 41, 42],
    "pierwszeństwo": [11, 12, 13, 14],
    "ostrzegawcze": list(range(18, 32)),
    "nakazy": list(range(33, 41)),
}

CLASS_GROUP = {cid: group for group, ids in _GROUP_RANGES.items() for cid in ids}

assert len(CLASS_NAMES_PL) == 43
assert sorted(CLASS_GROUP) == list(range(43))

_TRAIN_NAME = re.compile(r"(\d+)_(\d+)_(\d+)\.\w+$")


# --------------------------------------------------------------------------
# Funkcje
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Ustawia ziarno losowości (Python, NumPy i, jeśli jest, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Tworzy katalogi na wyniki, jeśli ich nie ma."""
    for d in (PROCESSED_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_data_dir(gtsrb_dir: Path | None = None) -> Path:
    """Sprawdza, czy w katalogu są pliki CSV i foldery z obrazami. Zwraca ścieżkę."""
    d = Path(gtsrb_dir or GTSRB_DIR)
    missing = [n for n in ("Train.csv", "Test.csv", "Meta.csv", "Train", "Test", "Meta") if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"W katalogu {d} brakuje: {', '.join(missing)}.\n"
            "Rozpakujcie zbiór GTSRB do data/GTSRB albo ustawcie zmienną środowiskową GTSRB_DIR."
        )
    return d


def _read_csv(path: Path, expected: list[str], dtype: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtype)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: brak kolumn {missing}; są: {list(df.columns)}")
    return df


def load_tables(gtsrb_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wczytuje Train.csv, Test.csv i Meta.csv i dodaje kolumny pomocnicze.

    Dodane kolumny (train i test): split, label (= ClassId), file (pełna ścieżka),
    roi_w, roi_h; tylko train: track (numer serii z nazwy pliku), frame, track_id.
    """
    d = check_data_dir(gtsrb_dir)
    train = _read_csv(d / "Train.csv", TRAIN_CSV_COLUMNS)
    test = _read_csv(d / "Test.csv", TRAIN_CSV_COLUMNS)
    # SignId to kod w stylu "4.10" – jako liczba straciłby końcowe zero (4.10 -> 4.1)
    meta = _read_csv(d / "Meta.csv", META_CSV_COLUMNS, dtype={"SignId": str})

    for split, df in (("train", train), ("test", test)):
        df.insert(0, "split", split)
        df["label"] = df["ClassId"].astype(int)
        df["file"] = [str(d / p) for p in df["Path"]]
        df["roi_w"] = df["Roi.X2"] - df["Roi.X1"]
        df["roi_h"] = df["Roi.Y2"] - df["Roi.Y1"]

    parsed = train["Path"].str.extract(_TRAIN_NAME).astype(float)
    train["track"] = parsed[1].astype("Int64")
    train["frame"] = parsed[2].astype("Int64")
    train["track_id"] = (train["label"].astype(str) + "_" + train["track"].astype(str)).where(train["track"].notna())

    meta["file"] = [str(d / p) for p in meta["Path"]]
    meta = meta.sort_values("ClassId").reset_index(drop=True)
    return train.reset_index(drop=True), test.reset_index(drop=True), meta


def load_image(path):
    """Otwiera obraz jako PIL RGB."""
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def to_array(img, size: int | None = 32, gray: bool = False) -> np.ndarray:
    """Zamienia obraz PIL na tablicę float32 w zakresie [0, 1].

    size=None zostawia oryginalny rozmiar; gray=True zwraca jeden kanał.
    """
    img = img.convert("L" if gray else "RGB")
    if size is not None:
        img = img.resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def savefig(fig, name: str) -> Path:
    """Zapisuje wykres do reports/figures/<name>.png (np. do prezentacji)."""
    ensure_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
