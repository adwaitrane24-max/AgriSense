# Viva notes: short answers to likely judge questions

Both of us should be able to say these in our own words.

## Machine learning

**What is transfer learning and why use it?**
We start from a network (MobileNetV2) that already learned general image features (edges,
textures, colours) from 1.4 million ImageNet photos. We freeze it and train only a small new
head for our 19 classes. With ~4,000 training images and a free GPU for 20 minutes, training a
CNN from scratch would be far less accurate.

**Why MobileNetV2?**
It is designed for phones: ~3.5 M parameters, ~10 MB file, ~300 M multiply-adds per image, so
it runs in ~100 ms on our laptop CPU. Bigger models (ResNet50, ~25 M parameters) are slower
and not needed for PlantVillage.

**What is overfitting and how did you check for it?**
Overfitting = the model memorises training images and does worse on new ones. We (1) keep a
validation set and plot train vs validation accuracy/loss (`reports/training_curves.png`),
(2) use EarlyStopping on validation loss with best weights restored, (3) use dropout 0.2 and
data augmentation, and (4) report final numbers only on a test set the model never saw.

**Why a stratified split?**
Each split keeps the same proportion of every class. Without it, a small class could end up
with very few test images by chance, and the test accuracy would be misleading.

**Why is accuracy alone not enough? What does the confusion matrix show?**
Accuracy is one number over all classes; it can hide one class that is always wrong.
Per-class precision/recall/F1 show each class. The confusion matrix shows which class gets
mistaken for which (rows = true, columns = predicted). Similar-looking diseases (for example
early vs late blight, both brown spots) are the usual confusions — check `reports/results.md`
for our actual top 2.

**What happens with a non-leaf or unknown-crop photo?**
The softmax always spreads 100 % over our 19 classes, so it must pick something. Usually the
top confidence is low, so the 0.60 threshold shows "Not sure". This reduces but does not solve
the problem: a model can be confidently wrong on something it never saw.

**Would it work in a real field?**
Less well. PlantVillage photos are single leaves on plain backgrounds under lab lighting.
Next steps: collect and label field photos, fine-tune on them, add a "is this a leaf?" check,
and test with real users together with experts.

## Data structures and complexity

**How does the LRU cache work and why is it O(1)?**
A dict maps key → node, and the nodes form a doubly linked list ordered from most to least
recently used, with dummy head and tail nodes. `get`: dict lookup O(1), unlink node and move it
behind head O(1). `put`: if full, remove the node before tail (the least recently used) and
delete its key from the dict, then insert the new node behind head. Every step is a fixed
number of pointer changes, independent of how many items are stored. Space is O(capacity).

**Why a doubly linked list, not a normal list?**
Removing an element from the middle of a Python list is O(n) because the rest shifts. With
prev/next pointers we can unlink any node in O(1).

**Why hash the content instead of the filename?**
Same image renamed → same hash → cache hit. Different images both called `leaf.jpg` → different
hashes → no wrong answer. SHA-256 is O(file size) and collisions are practically impossible.

**Time complexity of one prediction? Of top-k?**
Prediction: the network and input size are fixed, so it is constant work per image (~300 M
multiply-adds), ~100 ms on CPU. Preprocessing is O(pixels) of the upload. Top-k with
`heapq.nlargest` is O(n log k): it scans n = 19 scores and keeps a heap of size k = 3.

**What is the cache hit time?**
Validation + SHA-256 + dict lookup: a few milliseconds (shown on `/stats`), versus ~100+ ms
for a miss.

## Engineering

**How is the code organised?**
One job per module (see README table). `app.py` has routes only; the pipeline is in
`leafcheck/service.py`. All numbers (threshold, cache size, upload limit) are in `config.py`.

**How did you test without the model?**
`FakePredictor` has the same interface as the real one and returns fixed probabilities. It
also counts calls, so the API test proves that the second upload of the same image does not
run the model. Only `test_model_sanity.py` loads the real model and it is skipped if missing.

**How did you divide work and use Git?**
Person A: notebooks, predictor, preprocess, reports. Person B: cache, hashing, stats, app,
UI, API tests. Each feature on its own branch, merged into `main` by pull request after the
other person reviewed it. Small commits with clear messages.

**Ethics and safety?**
No treatment advice; every result has a disclaimer pointing to an expert / Krishi Vigyan
Kendra; low confidence shows "Not sure"; uploads are never stored; dataset is public and cited.
