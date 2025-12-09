import numpy as np
import pandas as pd
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# === Korean font setup ===
system_name = platform.system()
if system_name == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

print("Font setup complete!")


# ===============================================================
# 1. parquet 파일 자동 탐색 + globals() 로 데이터 로드
# ===============================================================
base_dir = Path("data")

folder_map = {
    "customer":   "1.회원정보",
    "credit":     "2.신용정보",
    "sales":      "3.승인매출정보",
    "billing":    "4.청구입금정보",
    "balance":    "5.잔액정보",
    "channel":    "6.채널정보",
    "marketing":  "7.마케팅정보",
    "performance":"8.성과정보",
}

info_categories = list(folder_map.keys())


def find_parquet_file(folder: Path):
    """해당 폴더 내에서 train*.parquet 파일을 자동 탐색"""
    files = list(folder.glob("train*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError(f"train*.parquet not found in {folder}")
    return files[0]


# --- parquet 파일 읽고 globals에 저장 ---
for prefix in info_categories:
    folder = base_dir / folder_map[prefix]
    file_path = find_parquet_file(folder)
    df = pd.read_parquet(file_path)
    globals()[f"{prefix}_train"] = df
    print(f"Loaded {prefix}_train: {df.shape}")

# ===============================================================
#  2. Concat monthly datasets (your structure supports multi-month)
# ===============================================================
train_dfs = {}
for prefix in info_categories:
    df_list = [globals()[f"{prefix}_train"]]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    print(f"{prefix}_train_df created: {train_dfs[f'{prefix}_train_df'].shape}")

customer_train_df    = train_dfs["customer_train_df"]
credit_train_df      = train_dfs["credit_train_df"]
sales_train_df       = train_dfs["sales_train_df"]
billing_train_df     = train_dfs["billing_train_df"]
balance_train_df     = train_dfs["balance_train_df"]
channel_train_df     = train_dfs["channel_train_df"]
marketing_train_df   = train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

gc.collect()

# ===============================================================
#  3. Merge all dataframes on ID
# ===============================================================
from functools import reduce

COMMON_ID = "ID"
CONFLICT_COL = "기준년월"

data_to_merge = [
    customer_train_df, credit_train_df, sales_train_df,
    billing_train_df, balance_train_df, channel_train_df,
    marketing_train_df, performance_train_df
]

processed_list = []
for df in data_to_merge:
    df2 = df.copy()
    if CONFLICT_COL in df2.columns and CONFLICT_COL != COMMON_ID:
        df2 = df2.drop(columns=[CONFLICT_COL])
    processed_list.append(df2)

merged_train_df = reduce(
    lambda left, right: pd.merge(left, right, on=COMMON_ID, how='left'),
    processed_list
)

print("Merge done:", merged_train_df.shape)

# ===============================================================
#  4. Feature separation (numerical vs categorical)
# ===============================================================
target_col = "Segment"
id_col = ["customer_id"]

features_df = merged_train_df.drop(columns=[target_col] + id_col, errors='ignore')

Discrimination_criteria = 30

initial_categorical = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
initial_numerical = features_df.select_dtypes(include=np.number).columns.tolist()

refined_categorical = initial_categorical.copy()
refined_numeric = []

for col in initial_numerical:
    if features_df[col].nunique() < Discrimination_criteria:
        refined_categorical.append(col)
    else:
        refined_numeric.append(col)

print("Numeric:", len(refined_numeric))
print("Categorical:", len(refined_categorical))

# ===============================================================
#  5. ANOVA (numerical) + Chi2 (categorical)
# ===============================================================
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2

y = merged_train_df[target_col]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- ANOVA numeric ---
X_num = features_df[refined_numeric].fillna(0)
f_scores, p_values = f_classif(X_num, y_encoded)

num_results_df = pd.DataFrame({
    "Feature": refined_numeric,
    "F_Score": f_scores,
    "P_Value": p_values
}).sort_values(by="F_Score", ascending=False)

# --- Chi2 categorical ---
X_cat = features_df[refined_categorical].astype(str).fillna("Missing")
encoder = OrdinalEncoder()
X_cat_encoded = encoder.fit_transform(X_cat)
chi_scores, p_vals_cat = chi2(X_cat_encoded, y_encoded)

cat_results_df = pd.DataFrame({
    "Feature": refined_categorical,
    "Chi2_Score": chi_scores,
    "P_Value": p_vals_cat
}).sort_values(by="Chi2_Score", ascending=False)

# --- Select top 50 each ---
TOP_N = 50
top_num_features = num_results_df["Feature"].head(TOP_N).tolist()
top_cat_features = cat_results_df["Feature"].head(TOP_N).tolist()
key_features = top_num_features + top_cat_features

# ===============================================================
#  6. Preprocessing: missing, outliers, log-transform, label-encoding
# ===============================================================
print("Final preprocessing...")

X = merged_train_df[key_features].copy()

# numeric
X[top_num_features] = X[top_num_features].fillna(0)
X[top_num_features] = X[top_num_features].clip(lower=0)
for col in top_num_features:
    p99 = X[col].quantile(0.99)
    X[col] = X[col].clip(upper=p99)
for col in top_num_features:
    X[col] = np.log1p(X[col])

# categorical
le2 = LabelEncoder()
for col in top_cat_features:
    X[col] = X[col].astype(str).fillna("Missing")
    X[col] = le2.fit_transform(X[col])

print("Preprocessing complete!")
print("X shape:", X.shape, "y:", y_encoded.shape)






from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import numpy as np


# ===============================================================
# 1. Train / Validation / Test Split (60 / 20 / 20)
# ===============================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,          # 0.25 * 0.80 = 0.20
    random_state=42,
    stratify=y_train_full
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)


# ===============================================================
# 2. Cross-validation 객체 생성
# ===============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ===============================================================
# 3. 공통 평가 함수
# ===============================================================
def eval_model(model, X, y, name):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    print(f"[{name}] Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    return acc, f1


# ===============================================================
# 4-1. XGBoost 모델 및 하이퍼파라미터 튜닝
# ===============================================================
xgb_params = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.1],
    "n_estimators": [300, 600, 1000],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0]
}

xgb_model = XGBClassifier(
    tree_method="hist",
    predictor="auto",
    objective="multi:softmax",
    num_class=len(np.unique(y_encoded)),
    random_state=42
)


xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("\nBest XGBoost params:", xgb_search.best_params_)


# ===============================================================
# 4-2. LightGBM 모델 및 하이퍼파라미터 튜닝
# ===============================================================
lgbm_params = {
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 5, 10],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [500, 1000, 1500]
}

lgbm_model = LGBMClassifier(
    objective="multiclass",
    num_class=5,
    boosting_type="gbdt",
    random_state=42
)


lgbm_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=lgbm_params,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_
print("\nBest LGBM params:", lgbm_search.best_params_)


# ===============================================================
# 4-3. CatBoost 모델 및 하이퍼파라미터 튜닝
# ===============================================================
cat_params = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.1],
    "iterations": [500, 1000, 1500],
    "l2_leaf_reg": [1, 3, 5, 7]
}

cat_model = CatBoostClassifier(
    loss_function="MultiClass",
    task_type="GPU",
    devices='0',
    random_seed=42,
    verbose=0
)

cat_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=cat_params,
    n_iter=15,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

cat_search.fit(X_train, y_train)
best_cat = cat_search.best_estimator_
print("\nBest CatBoost params:", cat_search.best_params_)


# ===============================================================
# 5. Validation 및 Test 성능 비교
# ===============================================================
print("\n====== VALIDATION PERFORMANCE ======")
eval_model(best_xgb, X_valid, y_valid, "XGBoost Valid")
eval_model(best_lgbm, X_valid, y_valid, "LightGBM Valid")
eval_model(best_cat, X_valid, y_valid, "CatBoost Valid")

print("\n====== FINAL TEST PERFORMANCE ======")
eval_model(best_xgb, X_test, y_test, "XGBoost Test")
eval_model(best_lgbm, X_test, y_test, "LightGBM Test")
eval_model(best_cat, X_test, y_test, "CatBoost Test")

