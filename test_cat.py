import numpy as np
import pandas as pd
from scipy import stats
import gc

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc

# 1. 운영체제에 따른 폰트 설정

system_name = platform.system()

if system_name == 'Windows':
    # 윈도우의 경우 '맑은 고딕' 설정
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()

    
    rc('font', family=font_name)
else:
    print("Unknown system... sorry")

# 2. 마이너스(-) 기호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False

print("한글 폰트 설정 완료!")

info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]
#### Train data ####

# 각 유형별로 월별 데이터를 합쳐서 새로운 변수에 저장
train_dfs = {}

for prefix in info_categories:
    # globals()에서 동적 변수명으로 데이터프레임들을 가져와 리스트에 저장
    category = f"{prefix}_train"
    df_list = [globals()[category]]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_train_df is created with shape: {train_dfs[f'{prefix}_train_df'].shape}")

customer_train_df = train_dfs["customer_train_df"]
credit_train_df   = train_dfs["credit_train_df"]
sales_train_df    = train_dfs["sales_train_df"]
billing_train_df  = train_dfs["billing_train_df"]
balance_train_df  = train_dfs["balance_train_df"]
channel_train_df  = train_dfs["channel_train_df"]
marketing_train_df= train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

gc.collect()

from functools import reduce

# 공통 ID 컬럼명
COMMON_ID = 'ID' 
# 삭제할 중복 컬럼명
CONFLICT_COL = '기준년월' 

# DF
data_to_merge = [
    customer_train_df,
    credit_train_df,
    sales_train_df,
    billing_train_df,
    balance_train_df,
    channel_train_df,
    marketing_train_df,
    performance_train_df
]

all_train_dfs_processed = []

print(f"--- Dropping '{CONFLICT_COL}' column before merge ---")

# 1. 각 DF에서 COMMON_ID가 아닌 '기준년월' 컬럼 삭제
for i, df in enumerate(data_to_merge):
    df_processed = df.copy()
    
    # 해당 컬럼이 존재하고, ID 컬럼이 아닌 경우에만 삭제
    if CONFLICT_COL in df_processed.columns and CONFLICT_COL != COMMON_ID:
        df_processed = df_processed.drop(columns=[CONFLICT_COL])
        print(f"Dropped '{CONFLICT_COL}' from DataFrame {i+1}")
    
    all_train_dfs_processed.append(df_processed)

# 2. '기준년월'이 삭제된 DF 리스트로 reduce merge 실행
print("\nStarting merge...")
merged_train_df = reduce(
    lambda left, right: pd.merge(left, right, on=COMMON_ID, how='left'), 
    all_train_dfs_processed
)

print(f"--- Merge successful! ---")
print(f"최종 병합된 데이터 Shape: {merged_train_df.shape}") # 열 단위로 병합된 데이터 

# (이산형, 수치형) 데이터로 나누기 
target_col = 'Segment'
id_col = ['customer_id']

features_df = merged_train_df.drop(columns=[target_col] + id_col, errors='ignore')

Discrimination_criteria = 30 # 30 개를 기준으로 이산형, 수치형으로 분류

# object, categorical은 미리 이산형으로 분류
initial_categorical = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
initial_numerical = features_df.select_dtypes(include=np.number).columns.tolist()

refined_numerical_features = []
refined_categorical_features = list(initial_categorical) 

# 수치형 피처들을 다시 검토
for col in initial_numerical:
    if features_df[col].nunique() < Discrimination_criteria:
        refined_categorical_features.append(col)
    else:
        refined_numerical_features.append(col)

print(f"--- 정제된 피처 분리 결과 (임계값: {Discrimination_criteria}) ---")
print(f"수치형 피처: {len(refined_numerical_features)}개")
print(f"범주형 피처: {len(refined_categorical_features)}개")    

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2
import warnings

# --- 0. 타겟 변수(y) 인코딩 ---
y = merged_train_df[target_col]

# 'Segment' (A~E)를 숫자(0~4)로 변환
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 1. 수치형 피처 (Numerical) → ANOVA F-test ---
print("\n--- 1. 수치형 피처 vs. 타겟 (ANOVA F-test) ---")

# 결측치를 0으로 단순 대체 (추후 더 나은 imputation 방법 고려)
X_num = features_df[refined_numerical_features].fillna(0)

# ANOVA F-test 실행
f_scores, p_values = f_classif(X_num, y_encoded)

# 결과를 DataFrame으로 정리
num_results_df = pd.DataFrame({
    'Feature': refined_numerical_features,
    'F_Score': f_scores,
    'P_Value': p_values
})

# P_Value가 낮은 순(연관성이 높은 순)으로 정렬
num_results_df = num_results_df.sort_values(by='P_Value', ascending=True)
num_results_df = num_results_df.sort_values(by='F_Score', ascending=False)


print("타겟과 연관성 높은 수치형 피처 TOP 50:")
print(num_results_df.head)


# --- 2. 범주형 피처 (Categorical) → Chi-Squared ---

print("\n--- 2. 범주형 피처 vs. 타겟 (Chi-Squared) ---")

# 전처리 : 모든 범주형 피처를 '문자열'로 변환하고 결측치를 'Missing'으로 대체
X_cat = features_df[refined_categorical_features].astype(str).fillna('Missing')

# 범주형 피처들을 숫자로 인코딩 (Chi-Squared는 숫자 입력만 받음)
encoder = OrdinalEncoder()
X_cat_encoded = encoder.fit_transform(X_cat)

# Chi-Squared 테스트 실행
chi_scores, p_values = chi2(X_cat_encoded, y_encoded)

# 결과를 DataFrame으로 정리
cat_results_df = pd.DataFrame({
    'Feature': refined_categorical_features,
    'Chi2_Score': chi_scores,
    'P_Value': p_values
})

# P_Value가 낮은 순(연관성이 높은 순)으로 정렬
cat_results_df = cat_results_df.sort_values(by='P_Value', ascending=True)
cat_results_df = cat_results_df.sort_values(by='Chi2_Score', ascending=False)

print("타겟과 연관성 높은 범주형 피처 TOP 50:")
print(cat_results_df.head)

# ---결측치 존재 여부 및 패턴---

TOP_N = 50
# 1. 수치형 피처 상위 N개 추출
top_num_features = num_results_df['Feature'].head(TOP_N).tolist()

# 2. 범주형 피처 상위 N개 추출
top_cat_features = cat_results_df['Feature'].head(TOP_N).tolist()

key_features = top_num_features + top_cat_features

# 1. 전체 데이터프레임의 결측치 개수 확인
total_missing = merged_train_df.isnull().sum()

# 2. 결측치가 많은 순서로 정렬
missing_count_sorted = total_missing.sort_values(ascending=False)
# print("--- 결측치 많은 순서 (상위 20개) ---")
# #print(missing_count_sorted.head(20)) # 나중에 결측치가 90% 이상이면 drop 하기 

# 3. 선별한 '중요 피처'들의 결측치 개수 확인
print("\n--- 주요 피처 결측치 현황 ---")
print(total_missing[key_features])
aaaa = total_missing[key_features]

# --- 이상치 검출  ---

def detect_outliers_iqr_safe(df, column):
    series = pd.to_numeric(df[column], errors='coerce')
    
    # 2. 결측치 제외하고 계산
    series = series.dropna()
    
    if len(series) == 0:
        return pd.DataFrame() # 데이터 없으면 빈 DF 반환

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 3. 이상치 필터링
    outliers = df.loc[(series < lower_bound) | (series > upper_bound)]
    
    # 결과 출력
    if len(outliers) > 0:
        print(f"[IQR] {column} 이상치 개수: {len(outliers)}개")
    
    return outliers


print("\n--- 주요 수치형 피처 이상치 검출 시작 ---")
for col in top_num_features[:50]: 
    detect_outliers_iqr_safe(merged_train_df, col)


print("\n--- 주요 범주형 피처 이상치 검출 시작 ---")
for col in top_cat_features[:50]: 
    detect_outliers_iqr_safe(merged_train_df, col)

# 특정 피처 하나만 보고 싶을 때:
# target_feature = '이용금액_R3M_신용체크' # (예시 컬럼명)
# detect_outliers_iqr_safe(merged_train_df, target_feature)

bbbb = pd.DataFrame(merged_train_df.columns)

print("전처리 : 데이터 수정 및 변환 시작\n")
# 최종 피처
final_features = top_num_features + top_cat_features # 수처형 50개 , 범주형 50개
X = merged_train_df[final_features].copy()

#수치형 데이터 전처리 
print("전처리 : 수치형 데이터 전처리 \n")
X[top_num_features] = X[top_num_features].fillna(0) # 결측치 처리 -> 0
X[top_num_features] = X[top_num_features].clip(lower=0) # 마이너스 값 제거 -> 0

# 극단치 캡핑 (상위 1% 초과 -> 상위 1%)
for col in top_num_features:
    p99 = X[col].quantile(0.99)
    X[col] = X[col].clip(upper=p99)

# 금액 데이터 등 분포를 정규분포에 가깝게 만듦 -> 로그 변환
for col in top_num_features:
    X[col] = np.log1p(X[col])

# 범주형 데이터 전처리
print("전처리 : 범주형 데이터 전처리")
X[top_cat_features] = X[top_cat_features].fillna('Missing')

# label encoding 적용
le = LabelEncoder()
for col in top_cat_features:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

# 확인용 
print(X.shape)
print(y_encoded.shape)



import numpy as np
import pandas as pd
import gc

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

print("==== 모델링 시작 ====")

# ---------------------------------------
# 1) 데이터 분할 (60 / 20 / 20)
# stratify로 클래스 비율 유지
# ---------------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)
# (전체 대비 0.25 → 20%)

print(f"Train set : {X_train.shape}")
print(f"Valid set : {X_valid.shape}")
print(f"Test  set : {X_test.shape}")

# ---------------------------------------
# 2) 공통 평가지표 함수
# ---------------------------------------
def evaluate(model, X, y, name="Model"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    print(f"[{name}] Accuracy: {acc:.4f},  Macro F1: {f1:.4f}")
    return acc, f1

# ---------------------------------------
# 3) LGBM 하이퍼파라미터 튜닝
# ---------------------------------------
print("\n==== LGBM Hyperparameter Optimization ====")

lgbm_param_grid = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [500, 1000, 1500],
    "max_depth": [-1, 5, 10, 15],
    "min_child_samples": [20, 40, 60],
}

lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_encoded)),
    random_state=42
)

# 5-Fold Stratifed CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgbm_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=lgbm_param_grid,
    n_iter=20,
    scoring='f1_macro',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

lgbm_search.fit(X_train, y_train)
print("Best LGBM params:", lgbm_search.best_params_)

best_lgbm = lgbm_search.best_estimator_

# ---------------------------------------
# 4) CatBoost 하이퍼파라미터 튜닝
# ---------------------------------------
print("\n==== CatBoost Hyperparameter Optimization ====")

cat_param_grid = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05],
    "iterations": [500, 1000, 1500],
    "l2_leaf_reg": [1, 3, 5, 7]
}

cat = CatBoostClassifier(
    loss_function='MultiClass',
    verbose=0,
    random_state=42
)

cat_search = RandomizedSearchCV(
    estimator=cat,
    param_distributions=cat_param_grid,
    n_iter=15,
    scoring='f1_macro',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

cat_search.fit(X_train, y_train)
print("Best CatBoost params:", cat_search.best_params_)

best_cat = cat_search.best_estimator_

# ---------------------------------------
# 5) 각 모델 검증용 성능 평가
# ---------------------------------------
print("\n==== Validation Performance ====")
evaluate(best_lgbm, X_valid, y_valid, "LGBM Valid")
evaluate(best_cat, X_valid, y_valid, "CatBoost Valid")

# ---------------------------------------
# 6) 최종 Test 성능 평가
# ---------------------------------------
print("\n==== Final Test Performance ====")
evaluate(best_lgbm, X_test, y_test, "LGBM Test")
evaluate(best_cat, X_test, y_test, "CatBoost Test")

print("\n==== 전체 시스템 완료 ====")




# # ==========================================
# # 1. 전체 데이터를 Train 80% / Test 20%로 분할
# # ==========================================
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from catboost import CatBoostClassifier, Pool
# import numpy as np

# print("\n===== 1단계: Train 80% / Test 20% 분할 =====")

# # Test는 최종 평가용 → 절대 재사용하지 않음
# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X,
#     y_encoded,
#     test_size=0.20,
#     random_state=42,
#     stratify=y_encoded
# )

# print("Train(80%) :", X_train_full.shape)
# print("Test(20%)  :", X_test.shape)


# # ==========================================
# # 2. Train(80%) 내부에서 4-Fold Cross Validation 수행
# # ==========================================

# print("\n===== 2단계: Train 80% 내부 4-Fold 교차검증 =====")

# kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# fold_acc = []
# fold_f1 = []

# for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
#     print(f"\n▶ Fold {fold+1} 시작")

#     X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
#     y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]

#     # CatBoost용 Pool
#     train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
#     valid_pool = Pool(X_val, y_val, cat_features=cat_features)

#     # 모델 정의
#     cv_model = CatBoostClassifier(
#         iterations=500,
#         depth=6,
#         learning_rate=0.1,
#         loss_function="MultiClass",
#         eval_metric="MultiClass",
#         random_seed=42,
#         verbose=False
#     )

#     # 학습
#     cv_model.fit(train_pool, eval_set=valid_pool)

#     # 예측
#     pred = cv_model.predict(X_val).ravel()

#     # 지표 계산
#     acc = accuracy_score(y_val, pred)
#     f1 = f1_score(y_val, pred, average="macro")

#     fold_acc.append(acc)
#     fold_f1.append(f1)

#     print(f"[Fold {fold+1}] Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")

# print("\n===== 4-Fold 교차검증 결과 =====")
# print(f"평균 Accuracy: {np.mean(fold_acc):.4f}")
# print(f"평균 F1-macro: {np.mean(fold_f1):.4f}")


# # ==========================================
# # 3. Train(80%) 전체로 다시 학습 → Test 20%로 최종 평가
# # ==========================================

# print("\n===== 3단계: Train 전체로 최종 모델 재학습 =====")

# final_train_pool = Pool(X_train_full, y_train_full, cat_features=cat_features)

# final_model = CatBoostClassifier(
#     iterations=500,
#     depth=6,
#     learning_rate=0.1,
#     loss_function="MultiClass",
#     eval_metric="MultiClass",
#     random_seed=42,
#     verbose=100
# )

# final_model.fit(final_train_pool)


# # ==========================================
# # 4. Test(20%)에서 최종 평가
# # ==========================================

# print("\n===== 4단계: Test(20%) 최종 성능 =====")

# y_test_pred = final_model.predict(X_test).ravel()

# acc_test = accuracy_score(y_test, y_test_pred)
# f1_test = f1_score(y_test, y_test_pred, average="macro")

# print(f"Final Test Accuracy : {acc_test:.4f}")
# print(f"Final Test F1-macro : {f1_test:.4f}")

# print("=== Final Test Classification Report ===")
# print(classification_report(y_test, y_test_pred))

