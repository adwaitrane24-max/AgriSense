# LeafCheck: Crop Leaf Disease Detection Prototype

This file is the project brief for Claude Code. Read all of it before writing any code.

## 1. Context

- Course: Mini Project (CS208), Sardar Patel Institute of Technology, CSE, Semester V, AY 2026-27 (End Semester hackathon).
- Assigned area: "analysing crop images for possible disease indicators".
- Team: 2 students, comfortable with Python and web dev, **new to machine learning**.
- Hardware: laptops only (CPU). Model training happens in **free Google Colab (GPU)**.
- **Deadline: 2026-10-30, 12:00 noon IST.** Scope must stay small and finishable.

Problem statement requirements we must satisfy:
- A focused, testable prototype for one defined problem, with a working implementation and relevant tests.
- Use sample, simulated or authorised data only (we use the public PlantVillage dataset).
- State assumptions and limitations.
- Report crop-analysis performance measures.
- No real agricultural treatment advice; that stays outside the prototype.

## 2. How we are judged (design every decision around this)

| Criterion | Marks | What the project must show |
| --- | --- | --- |
| P1 Problem understanding, core CS concepts | 15 | Clear problem, users, scope; hashing, caching, image processing, train/validation/test split |
| P2 Logical approach, data structures and algorithms, time and space complexity | 20 | Hand-written LRU cache (hash map + doubly linked list), heap-based top-k, Big-O documented for every step, reasons for model choice |
| P3 Implementation and code quality: correctness, modularity, readability, Git repo, branching, commits | 30 | One job per module, type hints, docstrings, tests, README, feature branches, pull requests, small clear commits |
| P4 Demonstration | 15 | Live upload of leaves, a repeat upload showing a cache hit, a low-confidence "not sure" case, tests passing |
| P5 Answering judges' questions | 20 | Both students can explain every file; keep code simple enough to explain |

Also rated (1 to 3): engineering knowledge, problem analysis, design, modern tool usage, engineer and society, environment and sustainability, ethics, teamwork, communication, project management, real-world application.

**Implication:** the ML model is the simplest part. Most marks come from clean engineering, data structures with complexity analysis, tests, Git practice, and the students being able to explain everything.

## 3. Rules for Claude Code while working on this repo

1. **Explain as you go.** The students are new to ML. After writing each file, explain in plain words what it does, why, and its time/space complexity. They must answer judges' questions themselves.
2. **Keep it simple.** Prefer readable code over clever code. No frameworks beyond those listed below.
3. **Write the LRU cache by hand** (dict + doubly linked list). Do not use `functools.lru_cache` or `OrderedDict` for the core cache; it is a graded data-structure component.
4. **Git discipline:** work on feature branches (names in section 9), small commits with clear imperative messages (e.g. `Add LRU cache with O(1) get and put`), merge to `main` through pull requests. Never commit the dataset.
5. **Every module gets tests** before it is merged. Most tests must run without the real model (use a fake predictor).
6. **Document complexity** in docstrings for cache, hashing, preprocessing, top-k and prediction.
7. **No treatment advice anywhere** in code, UI text or README (no pesticides, doses or remedies).
8. Do not add features outside section 4's scope. If time is short, cut stretch goals, never tests or README.

## 4. Scope

**Product:** a simple web page where a user uploads one leaf photo and gets:
- the most likely class (disease or healthy),
- the top 3 classes with confidence scores,
- "Not sure, please consult an agriculture expert" when top confidence is below 0.60,
- a "from cache" badge when the same image was seen before,
- a disclaimer on every result.

**Crops and classes (19, from PlantVillage colour images):**
- Tomato (10): Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellow leaf curl virus, Mosaic virus, Healthy
- Corn/maize (4): Cercospora leaf spot (gray leaf spot), Common rust, Northern leaf blight, Healthy
- Potato (3): Early blight, Late blight, Healthy
- Bell pepper (2): Bacterial spot, Healthy

**In scope:** single-leaf classification, top 3, confidence threshold, LRU cache, `/stats` page, evaluation report, automated tests, README.

**Out of scope:** treatment advice, other crops or diseases, whole-plant or field photos, live camera, user accounts, mobile app, cloud deployment, saving uploaded images to disk.

## 5. Dataset

- Source: PlantVillage (colour), via TensorFlow Datasets `plant_village` or Kaggle. Cite it in the README.
- Keep only the 19 classes above; cap at about 300 images per class (about 5,700 total).
- Split 70% train / 15% validation / 15% test, **stratified**, fixed random seed (42).
- Preprocess: resize to 224 x 224, apply `tf.keras.applications.mobilenet_v2.preprocess_input`.
- Augment training data only: random horizontal flip, small rotation, small brightness change.
- Keep `samples/` with about 10 demo images: test-split images, 2 phone photos of real leaves, 1 non-leaf photo.
- `.gitignore` the raw dataset and any Colab outputs except the final model and labels.

## 6. Model

- Transfer learning with **MobileNetV2** (ImageNet weights, `include_top=False`, base frozen).
- Head: GlobalAveragePooling2D, Dropout(0.2), Dense(19, softmax).
- Train in Colab GPU: Adam, categorical cross-entropy, about 10 epochs, EarlyStopping on validation loss (patience 3, restore best weights).
- Save `model/leafcheck_mobilenetv2.keras` and `model/labels.json` (ordered list of class names). Save a copy to Google Drive right after training (Colab sessions can disconnect).
- Stretch goal only if ahead of schedule: unfreeze the last ~20 layers, retrain with learning rate 1e-5 for a few epochs.
- Target: 90% or higher test accuracy. Expect lower accuracy on real field photos; say so.

## 7. Repository layout

```
leafcheck/
  CLAUDE.md
  README.md
  requirements.txt          # pinned versions
  .gitignore
  app.py                    # Flask app: routes only, no business logic
  leafcheck/
    __init__.py
    config.py               # threshold (0.60), cache capacity, max upload size, paths
    cache.py                # hand-written LRU cache
    hashing.py              # SHA-256 fingerprint of image bytes
    preprocess.py           # bytes -> validated, resized, normalised array
    predictor.py            # loads model + labels once; predict(); top_k(); apply_threshold()
    stats.py                # hit/miss counters, response-time tracking
  templates/index.html      # upload form, result card, disclaimer
  static/style.css
  model/
    leafcheck_mobilenetv2.keras
    labels.json
  notebooks/
    train.ipynb             # Colab training
    evaluate.ipynb          # metrics, confusion matrix, per-class report
  reports/
    confusion_matrix.png
    classification_report.txt
    results.md
  samples/                  # demo images
  tests/
    conftest.py             # fake predictor fixture, Flask test client
    test_cache.py
    test_hashing.py
    test_preprocess.py
    test_predictor_logic.py
    test_api.py
    test_model_sanity.py    # only test that loads the real model; skip if model file missing
```

## 8. Request flow and data structures

`POST /predict` does:
1. Validate: JPG/PNG only, under 5 MB, decodable; otherwise return a clear 400 error.
2. `key = sha256(image_bytes)`.
3. `cache.get(key)`: hit -> return stored result with `from_cache: true`.
4. Miss -> preprocess to (1, 224, 224, 3), run model -> 19 probabilities.
5. Top 3 with `heapq.nlargest(3, ...)`.
6. If top probability < 0.60 -> status "not_sure" with the expert message.
7. `cache.put(key, result)` (evict least recently used when full), record timing in stats, return JSON; page renders it.

Other routes: `GET /` (page), `GET /stats` (hits, misses, hit rate, average ms for hits vs misses, cache size/capacity), `GET /health`.

Complexity table (put this in README and in docstrings):

| Part | Data structure / algorithm | Time | Space |
| --- | --- | --- | --- |
| Cache get/put | Hash map + doubly linked list (LRU) | O(1) average | O(capacity) |
| Fingerprint | SHA-256 | O(file size) | O(1) |
| Preprocess | Resize + normalise | O(pixels) | O(pixels) |
| Prediction | Fixed CNN, fixed input size | Constant per image (~300M multiply-adds) | Model ~14 MB |
| Top-k of n classes | Heap (`heapq.nlargest`) | O(n log k) | O(k) |
| Label lookup | List indexed by class id | O(1) | O(n) |

## 9. Git workflow and team split

- `main` always runs. All work via feature branches and pull requests reviewed by the other student.
- Person A (model): `feature/training`, `feature/evaluation`, `feature/predictor` -> notebooks, `predictor.py`, `preprocess.py`, model sanity test, reports.
- Person B (app): `feature/lru-cache`, `feature/web-ui`, `feature/stats`, `feature/tests` -> `cache.py`, `hashing.py`, `stats.py`, `app.py`, templates, API tests.
- Both: `docs/readme` -> README, slides material.
- Person B builds against a fake predictor so neither waits for the other.
- Aim for 25+ meaningful commits across both students, with each student's commits visible.

## 10. Tests (pytest, run with `pytest -q`)

- Cache: get after put; LRU eviction order when full; get refreshes recency; size never exceeds capacity; updating an existing key.
- Hashing: same bytes -> same key; different bytes -> different key.
- Preprocess: any input size -> (1, 224, 224, 3); rejects non-images and oversize files.
- Predictor logic (fake probabilities): top 3 sorted and length 3; threshold below 0.60 -> "not_sure"; above -> class name.
- API: valid image -> 200 with result; empty upload / text file -> 400 with message; same image twice -> second response `from_cache: true`; `/stats` counts update.
- Model sanity: on a fixed small sample of test images, accuracy above 85% (skipped if model missing).

## 11. Metrics to report (in `reports/results.md` and README)

| Metric | Target |
| --- | --- |
| Test accuracy (held-out 15%) | >= 90% |
| Precision, recall, F1 per class (`classification_report`) | Flag any class < 0.80 |
| Confusion matrix image | Explain top 2 confusions |
| "Not sure" rate and accuracy on confident predictions | Report both |
| Prediction time, cache miss (laptop CPU) | < 1 s |
| Prediction time, cache hit | a few ms |
| Cache hit rate during demo | Shown on `/stats` |

## 12. Limitations, safety and ethics (README section + one slide)

- Trained on lab photos (single leaf, plain background); field accuracy will be lower.
- Only 19 classes; unknown crops/diseases are forced into the nearest class (threshold mitigates, does not solve).
- High confidence is not proof; no severity estimate; tested only by the team.
- Small subset and some class imbalance.
- **Every result shows:** "This is an automated guess for learning purposes. Confirm with a local agriculture expert or Krishi Vigyan Kendra before taking any action."
- Uploaded images are processed in memory and not stored.
- Sustainability angle: early, expert-confirmed detection can reduce blanket spraying.

## 13. README must contain

Problem and users; features; screenshots; architecture and request flow; data structures and complexity table; dataset and citation; model and training summary; results table and confusion matrix; how to install and run (`pip install -r requirements.txt`, `flask --app app run`, `pytest -q`); limitations and safety; team and who did what.

## 14. Action plan (IST)

| Time | Person A (model) | Person B (app) |
| --- | --- | --- |
| 23:30 to 01:00 | Colab: load PlantVillage, filter 19 classes, cap 300 each, stratified split | Create repo, `.gitignore`, `requirements.txt`, Flask skeleton, `cache.py` + tests |
| 01:00 to 02:30 | Train MobileNetV2, evaluate, save model + labels to Drive and repo, push reports | Upload page, `/predict` with fake predictor, hashing, API tests, PRs reviewed |
| **Gate 02:30** | Model file and results pushed | |
| 02:30 to 07:30 | Sleep | Sleep |
| 07:30 to 09:30 | `predictor.py` with real model, sanity test, timing numbers | Top 3, threshold, disclaimer, `/stats`, cache-hit demo |
| **Gate 09:30** | Full demo works on both laptops | |
| 09:30 to 11:15 | Results in README, slides on data/model/metrics | Screenshots, slides on app/cache/tests/Git, final merge |
| 11:15 to 12:00 | Run demo twice, practise judge questions, submit | same |

If something slips: drop to 150 images per class; if not integrated by 09:30, freeze features and spend the time on demo, README and Q&A.

## 15. Demo script (about 5 minutes)

1. Problem and users (30 s).
2. Upload a tomato early blight leaf -> top 3 + disclaimer.
3. Upload a healthy potato leaf.
4. Upload the same image again -> "from cache", show `/stats` hit vs miss times.
5. Upload the non-leaf photo -> "Not sure" message.
6. Show confusion matrix and per-class results.
7. Run `pytest -q` live; show GitHub branches, PRs and commit history.
8. Limitations and safety.

## 16. Questions the students must be able to answer

- What is transfer learning and why use it? Why MobileNetV2?
- How does the LRU cache work and why is it O(1)? Why hash content instead of the filename?
- Time complexity of one prediction? Of top-k?
- What is overfitting and how did you check for it? Why stratified split?
- Why is accuracy alone not enough? What does the confusion matrix show?
- What happens with a non-leaf or unknown-crop photo?
- Would it work in a real field? What would you do next?
- How did you divide work and use Git?
