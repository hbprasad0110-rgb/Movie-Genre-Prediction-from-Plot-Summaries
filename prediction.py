import pandas as pd
import numpy as np
import re

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier


# =========================
# 1) Load data
# =========================
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
genres = pd.read_csv("movies_genres.csv")

valid_ids = set(genres["id"].astype(int).tolist())

# =========================
# 2) Parse labels robustly
# =========================
def parse_genres_safe(x):
    nums = re.findall(r"\d+", str(x))
    ids = [int(n) for n in nums]
    ids = [g for g in ids if g in valid_ids]
    return sorted(set(ids))

train["genre_list"] = train["genre_ids"].apply(parse_genres_safe)
train = train[train["genre_list"].apply(len) > 0].reset_index(drop=True)

print("Train rows:", len(train))
print("Avg labels/movie:", np.mean(train["genre_list"].apply(len)))

# =========================
# 3) Text
# =========================
train_text = (train["overview"].fillna("") + " " + train["title"].fillna("")).str.strip()
test_text  = (test["overview"].fillna("")  + " " + test["title"].fillna("")).str.strip()

# =========================
# 4) Labels (TMDB IDs)
# IMPORTANT: build classes from TRAIN to match competition label space
# =========================
classes = sorted(set(g for row in train["genre_list"] for g in row))
mlb = MultiLabelBinarizer(classes=classes)
y = mlb.fit_transform(train["genre_list"]).astype(np.int8)

print("Num labels:", y.shape[1])

# =========================
# 5) TF-IDF Features (word + char)
# =========================
word_vec = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=80000,
    min_df=2
)

Xw = word_vec.fit_transform(train_text)
Xc = char_vec.fit_transform(train_text)
X = hstack([Xw, Xc]).tocsr()

Xw_test = word_vec.transform(test_text)
Xc_test = char_vec.transform(test_text)
X_test = hstack([Xw_test, Xc_test]).tocsr()

# =========================
# 6) Split for threshold tuning
# =========================
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 7) Train one LightGBM model per label + collect probabilities
# =========================
n_labels = y.shape[1]
val_probs = np.zeros((X_val.shape[0], n_labels), dtype=np.float32)

# LightGBM parameters: good starting point for sparse TF-IDF
lgb_params = dict(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.5,
    reg_lambda=2.0,
    reg_alpha=0.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)

models = []
for j in range(n_labels):
    clf = LGBMClassifier(
        objective="binary",
        is_unbalance=True,   # handles imbalance automatically
        **lgb_params
    )
    clf.fit(
        X_tr, y_tr[:, j],
        eval_set=[(X_val, y_val[:, j])],
        eval_metric="binary_logloss",
        # verbose=False
    )
    val_probs[:, j] = clf.predict_proba(X_val)[:, 1]
    models.append(clf)

print("Finished training per-label LightGBM models.")

# =========================
# 8) Per-label threshold tuning (macro-F1)
# =========================
grid = np.arange(0.05, 0.60, 0.05)
thresholds = np.zeros(n_labels, dtype=np.float32)

for j in range(n_labels):
    best_t, best_f = 0.2, -1
    for t in grid:
        pred_j = (val_probs[:, j] >= t).astype(int)
        f = f1_score(y_val[:, j], pred_j, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    thresholds[j] = best_t

val_pred = (val_probs >= thresholds).astype(int)

# Ensure at least 1 label per movie (top-1)
for i in range(val_pred.shape[0]):
    if val_pred[i].sum() == 0:
        val_pred[i, np.argmax(val_probs[i])] = 1

val_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
print("Validation Macro F1:", val_macro)

# Optional: cap max labels to protect precision (macro-F1)
MAX_K = 4
for i in range(val_pred.shape[0]):
    if val_pred[i].sum() > MAX_K:
        top = np.argsort(-val_probs[i])[:MAX_K]
        tmp = np.zeros(n_labels, dtype=int)
        tmp[top] = 1
        val_pred[i] = tmp

val_macro_capped = f1_score(y_val, val_pred, average="macro", zero_division=0)
print("Validation Macro F1 (capped):", val_macro_capped)

# =========================
# 9) Refit on FULL train for final prediction
# =========================
test_probs = np.zeros((X_test.shape[0], n_labels), dtype=np.float32)

final_models = []
for j in range(n_labels):
    clf = LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        **lgb_params
    )
    clf.fit(X, y[:, j])
    test_probs[:, j] = clf.predict_proba(X_test)[:, 1]
    final_models.append(clf)

# Apply thresholds
test_pred = (test_probs >= thresholds).astype(int)

# Ensure at least 1 label and cap max labels
for i in range(test_pred.shape[0]):
    if test_pred[i].sum() == 0:
        test_pred[i, np.argmax(test_probs[i])] = 1
    if test_pred[i].sum() > MAX_K:
        top = np.argsort(-test_probs[i])[:MAX_K]
        tmp = np.zeros(n_labels, dtype=int)
        tmp[top] = 1
        test_pred[i] = tmp

# =========================
# 10) Build submission (space-separated genre IDs)
# =========================
def binarized_to_labels(row):
    labels = [str(mlb.classes_[i]) for i, v in enumerate(row) if v == 1]
    return " ".join(labels)

submission = pd.DataFrame({
    "movie_id": test["movie_id"].astype(int),
    "genre_ids": [binarized_to_labels(r) for r in test_pred]
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Avg predicted labels/movie:", np.mean(submission["genre_ids"].apply(lambda s: len(str(s).split()))))
print(submission.head(10))
