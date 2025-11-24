import numpy as np
import pandas as pd
from scipy import stats
import gc

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 데이터 분할(폴더) 구분
data_splits = ["train", "test"]

# 각 데이터 유형별 폴더명, 파일 접미사, 변수 접두어 설정
data_categories = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}

for split in data_splits:
    for category, info in data_categories.items():
        folder = info["folder"]
        suffix = info["suffix"]
        var_prefix = info["var_prefix"]
        
        file_path = f"./data/{folder}/{split}_{suffix}.parquet"
        # 변수명 형식: {var_prefix}_{split}_{month}
        variable_name = f"{var_prefix}_{split}"
        globals()[variable_name] = pd.read_parquet(file_path, engine="pyarrow")
        print(f"{variable_name} is loaded from {file_path}")

gc.collect()

# 데이터 유형별 설정 
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]

#### Train data ####

# 각 유형별로 월별 데이터를 합쳐서 새로운 변수에 저장
train_dfs = {}

for prefix in info_categories:
    # globals()에서 동적 변수명으로 데이터프레임들을 가져와 리스트에 저장
    df_list = [globals()[f"{prefix}_train"]]
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

# (★) 모든 범주형 피처를 '문자열'로 변환하고 결측치를 'Missing'으로 대체
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

print('수치형 데이터의 높은 연관성 : (청구원금, 이용금액)')
print('이산형 데이터의 높은 연관성 : (온라인 이용, 페이(간편결제) 및 할부)')

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
for col in top_num_features[:30]: 
    detect_outliers_iqr_safe(merged_train_df, col)


print("\n--- 주요 범주형 피처 이상치 검출 시작 ---")
for col in top_cat_features[:30]: 
    detect_outliers_iqr_safe(merged_train_df, col)

# 특정 피처 하나만 보고 싶을 때:
# target_feature = '이용금액_R3M_신용체크' # (예시 컬럼명)
# detect_outliers_iqr_safe(merged_train_df, target_feature)

bbbb = pd.DataFrame(merged_train_df.columns)