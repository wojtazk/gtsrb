# Odporność rozpoznawania znaków drogowych na zakłócenia i ataki (GTSRB)

Projekt z kursu **Metody AI w badaniu zagrożeń w systemach komputerowych**.
Zespół: [Imię Nazwisko], [Imię Nazwisko], [Imię Nazwisko]

Sprawdzamy, jak klasyfikatory znaków drogowych trenowane na zbiorze GTSRB radzą sobie z naturalnymi zakłóceniami (szum, rozmycie, kompresja…) i z atakami adwersarialnymi (FGSM, PGD, ataki celowane), które pomyłki są groźne dla bezpieczeństwa ruchu i czy zaburzone wejścia da się wykryć metodami wykrywania anomalii.

Projekt prowadzimy według metodyki **CRISP-DM**: każdy etap ma swój notatnik, a historia pracy jest w Git (commity, pull requesty i tagi na kamienie milowe).

---

## Struktura repozytorium

```
gtsrb-robustness/
├── README.md                ← ten plik
├── requirements.txt         ← biblioteki (instalacja jedną komendą)
├── .gitignore               ← dane i wagi modeli nie trafiają do repo
├── src/
│   └── gtsrb_utils.py       ← wspólny kod: nazwy klas po polsku, grupy znaków, ścieżki, ładowanie danych, ziarno
├── notebooks/
│   ├── 01_business_and_data_understanding.ipynb   ← CRISP-DM 1–2 (gotowe)
│   ├── 02_data_preparation.ipynb                  ← CRISP-DM 3 (do zrobienia)
│   └── …                                          ← kolejne etapy, plan niżej
├── data/
│   ├── GTSRB/               ← rozpakowany zbiór danych (każdy trzyma lokalnie, nie w repo)
│   └── processed/           ← pliki pośrednie generowane przez notatniki (nie w repo)
├── models/                  ← zapisane wagi modeli (nie w repo)
└── reports/
    └── figures/             ← wykresy generowane przez notatniki (do raportu i prezentacji)
```

Zasada: **logika wspólna dla wielu notatników trafia do `src/`**, a notatnik zawiera opis, wywołania i wnioski. Dzięki temu np. nazwy klas są zdefiniowane w jednym miejscu, a zmiana w nich widać w historii jednego pliku.

---

## Szybki start

### Lokalnie (zalecane do etapów 1–3, wystarczy CPU)

```bash
git clone https://github.com/<organizacja>/gtsrb-robustness.git
cd gtsrb-robustness
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

W Jupyter Lab otwórzcie `notebooks/01_business_and_data_understanding.ipynb` i wybierzcie **Kernel → Restart Kernel and Run All Cells**.

### Dane

Zbiór GTSRB (Zenodo: https://zenodo.org/records/13741936, plik `data.zip`, albo ta sama wersja z Kaggle) rozpakujcie tak, żeby powstał katalog `data/GTSRB/`:

```
data/GTSRB/
├── Train/0 … Train/42/    obrazy treningowe, nazwa pliku: <klasa>_<seria>_<klatka>.png
├── Test/                  obrazy testowe
├── Meta/                  wzorcowy obraz każdej klasy
├── Train.csv  Test.csv  Meta.csv
```

- **Foldery pisane małą literą** (`train/`, `test/`, `meta/`) są w archiwum kopiami tych z wielkiej litery. Notatnik sprawdza to w sekcji 2.1 i z nich nie korzysta. Gdy potwierdzi, że to kopie, możecie je usunąć – oszczędzicie mniej więcej połowę miejsca.
- **Dane leżą gdzie indziej?** Ustawcie zmienną środowiskową `GTSRB_DIR`, np. `export GTSRB_DIR=/home/ja/datasets/GTSRB` przed `jupyter lab` (Windows: `set GTSRB_DIR=D:\datasets\GTSRB`).
- Danych **nie dodajemy do repozytorium** – `data/GTSRB/` jest w `.gitignore`.

### Google Colab (gdy potrzebne GPU, od notatnika 03)

Wrzućcie spakowany zbiór (`GTSRB.zip`) na swój Google Drive, a w pierwszej komórce notatnika w Colabie:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/<organizacja>/gtsrb-robustness.git
%cd gtsrb-robustness/notebooks
!pip install -q -r ../requirements.txt
!unzip -q /content/drive/MyDrive/GTSRB.zip -d ../data/     # powstaje ../data/GTSRB
```

Rozpakowanie na dysk Colaba jest dużo szybsze niż czytanie tysięcy małych plików bezpośrednio z Google Drive. Dla prywatnego repozytorium potrzebny jest token GitHub (Settings → Developer settings → Personal access tokens) w adresie: `https://<token>@github.com/...`. Nie zapisujcie tokenu w notatniku, który commitujecie.

---

## Notatnik 01 – Business Understanding i Data Understanding

**Plik:** `notebooks/01_business_and_data_understanding.ipynb`
**Czas działania:** kilka minut (odczyt ok. 50 tys. obrazów). GPU niepotrzebne.
**Wejście:** `data/GTSRB/`. **Wyjście:** wykresy w `reports/figures/01_*.png` i tabela `data/processed/meta.csv` (metadane każdego obrazu z numerem serii i ramką ROI).

| Sekcja | Co zawiera |
|---|---|
| 0. Konfiguracja | importy, ziarno losowości, wersje bibliotek (dla powtarzalności) |
| 1.1 Cele biznesowe | kontekst TSR/ADAS/ISA, problem, cel projektu, interesariusze |
| 1.2 Ocena sytuacji | zasoby, wymagania kursu, założenia, ograniczenia, ryzyka z planem awaryjnym, słowniczek |
| 1.3 Model zagrożeń | kto, czym i po co atakuje klasyfikator; tabela kosztu pomyłek |
| 1.4 Cele analityczne | przełożenie celu biznesowego na zadania ML (A1–A7), metryki, kryteria sukcesu |
| 1.5 Plan projektu | etapy CRISP-DM ↔ notatniki ↔ kroki z kursu ↔ kamienie milowe |
| 2.1 Pozyskanie danych | inwentarz katalogu (pliki, rozmiar), **sprawdzenie, czy foldery małą literą są kopiami**, wczytanie trzech plików CSV |
| 2.2 Opis danych | znaczenie kolumn CSV, statystyki rozmiarów i ramek ROI, **siatka obrazów wzorcowych (Meta) do kontroli polskich nazw klas**, kody kształtu i koloru, metadane każdego obrazu (format, rozmiar, jasność, kontrast, skrót MD5) |
| 2.3 Eksploracja | rozkład klas i niezbalansowanie, trening vs test, przykłady z każdej klasy, **serie zdjęć tego samego znaku**, rozmiary, udział znaku w obrazie (ROI), jasność i kontrast, średni obraz klasy, PCA |
| 2.4 Jakość danych | spójność CSV ↔ pliki na dysku, pliki nieczytelne, zgodność folderu i nazwy pliku z klasą, poprawność ROI, duplikaty, przeciek trening↔test, spójność serii, zapis `meta.csv` |
| 2.5 Podsumowanie | tabela kluczowych liczb, wnioski do uzupełnienia, decyzje dla Data Preparation, literatura |

### Co musicie uzupełnić sami

- **Pola w nawiasach kwadratowych** `[...]`: skład zespołu, sprzęt, daty kamieni milowych, próg w kryterium A4.
- **Sekcja 2.5 „Wnioski z eksploracji”**: wpiszcie liczby z tabeli podsumowania i własne obserwacje z wykresów. Notatnik liczy wszystko sam, ale interpretacja należy do was.
- **Siatka obrazów wzorcowych w sekcji 2.2**: przejrzyjcie, czy polski podpis pasuje do rysunku znaku w każdym polu. Jeśli nie, poprawcie `CLASS_NAMES_PL` w `src/gtsrb_utils.py`.
- **Progi w kryteriach sukcesu (1.4)** są naszą propozycją – warto je potwierdzić z prowadzącym.

### Na co zwrócić uwagę w wynikach

- **Serie zdjęć (2.3).** Nazwa pliku treningowego `<klasa>_<seria>_<klatka>.png` mówi, które zdjęcia to ten sam fizyczny znak. Notatnik zapisuje to jako `track_id`. Walidację trzeba wydzielać po seriach – inaczej prawie identyczne klatki trafią do treningu i walidacji, a wynik będzie zawyżony.
- **Tabela jakości (2.4).** Każdy wiersz powinien mieć 0 (poza liczbą małych obrazów). Wartość różna od zera to rzecz do opisania w raporcie i do obsłużenia w notatniku 02.

---

## Plan kolejnych notatników

| Notatnik | Etap CRISP-DM | Zakres |
|---|---|---|
| `02_data_preparation` | 3. Data Preparation | skalowanie, wycięcie znaku po ROI, normalizacja, podział po seriach, wagi klas, augmentacja bez odbić lustrzanych |
| `03_baseline_models` | 4. Modeling | CNN od zera, transfer learning, model klasyczny na cechach (np. HOG + SVM); czasy uczenia i inferencji |
| `04_robustness_corruptions` | 4. Modeling | zakłócenia generowane w 5 poziomach siły (szum, rozmycie, mgła, jasność, JPEG, okluzja), tylko na zbiorze testowym |
| `05_adversarial_attacks` | 4. Modeling | FGSM, PGD, ataki celowane, trening adwersarialny |
| `06_anomaly_detection` | 4. Modeling | wykrywanie zaburzonych wejść: kNN / LOF / Isolation Forest na cechach sieci, AUROC |
| `07_evaluation_interpretability` | 5. Evaluation | porównanie modeli i literatury, Grad-CAM, pomyłki krytyczne |
| `08_demo` *(opcjonalnie)* | 6. Deployment | prosta demonstracja, test na innym zbiorze znaków (szansa na 5.5) |

Każdy notatnik zaczyna się od tej samej komórki konfiguracji (`sys.path.append("../src")`, `gu.set_seed()`), czyta wyniki poprzedniego z `data/processed/` i kończy się sekcją wniosków.

---

## Historia kodu: jak pracujemy z Git

Zasady kursu wymagają kamieni milowych co 2 tygodnie i reprodukowalnego kodu. Repozytorium na GitHubie załatwia oba punkty: prowadzący widzi postęp w historii commitów, a każdy kamień milowy jest oznaczony tagiem.

### Pierwsze uruchomienie (robi jedna osoba)

1. Załóżcie puste repozytorium na GitHubie (np. `gtsrb-robustness`) i dodajcie resztę zespołu w **Settings → Collaborators**. Warto też dodać prowadzącego, jeśli o to poprosi.
2. W katalogu z tymi plikami:

```bash
git init
git add .
git commit -m "Szkielet projektu: struktura katalogów, README, requirements"
git branch -M main
git remote add origin https://github.com/<organizacja>/gtsrb-robustness.git
git push -u origin main
```

3. Po uruchomieniu notatnika 01 na pełnych danych i uzupełnieniu wniosków:

```bash
git add notebooks/01_business_and_data_understanding.ipynb reports/figures/01_*.png
git commit -m "BU+DU: wyniki eksploracji GTSRB i wnioski"
git tag -a M1 -m "Kamień milowy 1: Business i Data Understanding"
git push --follow-tags
```

### Codzienna praca

- **Jedna gałąź na jedno zadanie**, np. `git switch -c du/analiza-serii`. Po skończeniu: push i **pull request** na GitHubie, druga osoba z zespołu go przegląda i zatwierdza. Widać wtedy, kto co zrobił i że zespół sprawdza nawzajem swoją pracę.
- **Małe commity z opisem**, z prefiksem etapu CRISP-DM:
  `DU: dodaj wykrywanie duplikatów`, `DP: podział po seriach`, `MOD: CNN baseline 32x32`, `EVAL: Grad-CAM dla pomyłek krytycznych`.
- **Przed commitem notatnika: Restart Kernel and Run All Cells.** Wyniki w repozytorium są wtedy spójne z kodem, a prowadzący widzi wykresy bezpośrednio na GitHubie.
- **Czytelne różnice w notatnikach:** pliki `.ipynb` to JSON, więc zwykły `git diff` jest nieczytelny. Po instalacji wymagań uruchomcie raz `nbdime config-git --enable --global` – od tej pory `git diff` pokazuje zmiany komórka po komórce.
- **Nie commitujcie danych, wag modeli ani tokenów** – `.gitignore` już to blokuje dla typowych plików.
- **Tag na każdy kamień milowy** (`M1`, `M2`, …): prowadzący może otworzyć stan projektu z dowolnego spotkania.

### Opcjonalnie

- **GitHub Issues** na zadania i **Projects** (tablica kanban) na podział pracy w zespole.
- Do raportu końcowego: link do repozytorium z tagiem `final` jako „załączony kod eksperymentów”.

---

## Źródła

- Stallkamp J., Schlipsing M., Salmen J., Igel C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks 32.
- Zbiór na Zenodo: https://zenodo.org/records/13741936
- Hendrycks D., Dietterich T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR.
- Chapman P. i in. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.*
