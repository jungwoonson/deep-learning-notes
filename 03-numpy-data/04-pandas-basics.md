# Pandas 기초 (Pandas Basics)

## 왜 필요한가: VLA로 가는 길

VLA 모델을 학습시키려면 대량의 데이터를 체계적으로 관리해야 한다. 현실의 데이터는 깔끔한 NumPy 배열이 아니라 **지저분한 표(table)** 형태로 존재한다.

- **Vision**: 이미지 파일 경로, 라벨, bounding box 좌표가 담긴 CSV
- **Language**: 텍스트와 메타데이터(언어, 출처, 품질 점수)가 담긴 테이블
- **Action**: 로봇 에피소드별 상태, 행동, 보상이 기록된 로그 파일
- **데이터셋 관리**: 학습/검증/테스트 분할, 클래스별 통계, 결측치 처리

Pandas는 이런 **표 형태 데이터를 불러오고, 탐색하고, 정리하는 표준 도구**다. 모델 학습 전 데이터를 이해하고 전처리하는 단계에서 반드시 필요하다.

---

## 핵심 개념

### 1. DataFrame과 Series

Pandas의 두 가지 핵심 자료구조:

- **Series**: 1차원 데이터 (인덱스가 붙은 배열). 하나의 열(column)에 해당.
- **DataFrame**: 2차원 데이터 (표). 여러 Series의 모음.

```
# Series
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# DataFrame
df = pd.DataFrame({
    'image_path': ['img001.jpg', 'img002.jpg', 'img003.jpg'],
    'label': ['cat', 'dog', 'cat'],
    'width': [224, 256, 224]
})
```

DataFrame은 **엑셀 시트**와 비슷하다. 각 열은 이름이 있고, 각 행은 인덱스가 있다. 다만 수십만~수백만 행도 빠르게 처리할 수 있다.

### 2. CSV 파일 읽기/쓰기

실무에서 가장 많이 하는 작업이다.

```
# 읽기
df = pd.read_csv('dataset/annotations.csv')
df = pd.read_csv('data.tsv', sep='\t')           # 탭 구분
df = pd.read_csv('data.csv', encoding='utf-8')    # 인코딩 지정

# 쓰기
df.to_csv('output.csv', index=False)
```

다른 형식도 지원한다:

```
pd.read_json('data.json')
pd.read_parquet('data.parquet')     # 대용량 데이터에 효율적
pd.read_excel('data.xlsx')
```

**VLA 연결**: 대규모 로봇 데이터셋(예: Open X-Embodiment)의 메타데이터는 CSV나 Parquet으로 관리된다. 어떤 에피소드가 어떤 태스크인지, 성공/실패 여부 등의 정보를 Pandas로 탐색한다.

### 3. 데이터 탐색 (Exploration)

데이터를 불러온 직후 **반드시** 해야 할 탐색 작업들:

#### 기본 확인

```
df.head()              # 처음 5행
df.tail()              # 마지막 5행
df.shape               # (행 수, 열 수)
df.columns             # 열 이름 목록
df.dtypes              # 각 열의 데이터 타입
df.info()              # 종합 정보 (타입, 결측치 수 등)
```

#### 통계 요약

```
df.describe()          # 수치형 열의 통계 (평균, 표준편차, 최솟값, 최댓값 등)
```

`describe()`가 알려주는 것:
- **count**: 결측치가 아닌 값의 수 (전체보다 적으면 결측치 존재)
- **mean, std**: 평균과 표준편차 (값의 스케일과 분포 파악)
- **min, 25%, 50%, 75%, max**: 분포의 형태 파악 (이상치 존재 여부)

#### 범주형 데이터 탐색

```
df['label'].value_counts()           # 각 값의 빈도
df['label'].nunique()                # 고유값의 수
df['label'].unique()                 # 고유값 목록
```

**VLA 연결**: 학습 데이터의 클래스 분포를 확인하는 것은 매우 중요하다. 클래스 불균형(class imbalance)이 심하면 모델이 다수 클래스에 편향된다. `value_counts()`로 이를 바로 확인할 수 있다.

### 4. 데이터 선택과 필터링

#### 열 선택

```
df['label']                         # 단일 열 → Series
df[['image_path', 'label']]         # 복수 열 → DataFrame
```

#### 행 필터링 (Boolean Indexing)

NumPy의 boolean indexing과 같은 원리다.

```
df[df['label'] == 'cat']                        # label이 'cat'인 행만
df[df['width'] > 200]                            # width가 200 초과인 행만
df[(df['label'] == 'cat') & (df['width'] > 200)] # 두 조건 동시 만족
```

#### loc와 iloc

```
df.loc[0:5, 'label']              # 이름 기반: 0~5행, 'label' 열
df.iloc[0:5, 1]                   # 위치 기반: 0~4행, 1번째 열
```

### 5. 기본 전처리 (Preprocessing)

#### 결측치 처리

```
df.isnull().sum()                  # 각 열의 결측치 수
df.dropna()                        # 결측치가 있는 행 제거
df.fillna(0)                       # 결측치를 0으로 채우기
df['col'].fillna(df['col'].mean()) # 평균값으로 채우기
```

#### 데이터 타입 변환

```
df['width'] = df['width'].astype(float)
df['label_code'] = df['label'].astype('category').cat.codes
```

#### 새 열 추가

```
df['area'] = df['width'] * df['height']
df['is_large'] = df['area'] > 50000
```

#### 정렬과 그룹화

```
df.sort_values('width', ascending=False)       # width 기준 내림차순 정렬

df.groupby('label')['width'].mean()            # label별 평균 width
df.groupby('label').size()                     # label별 데이터 수
```

**VLA 연결**: 데이터 전처리는 모델 성능에 직접적인 영향을 미친다. 결측치가 있는 에피소드를 걸러내고, 라벨을 숫자로 인코딩하고, 태스크별 통계를 확인하는 것이 학습 시작 전의 필수 과정이다.

### 6. 학습/검증/테스트 분할 준비

Pandas로 데이터를 탐색한 후 학습에 사용할 분할을 만드는 패턴:

```
# 셔플
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 비율로 분할
n = len(df_shuffled)
train_df = df_shuffled[:int(0.8 * n)]
val_df = df_shuffled[int(0.8 * n):int(0.9 * n)]
test_df = df_shuffled[int(0.9 * n):]
```

또는 `sklearn`의 `train_test_split`을 사용하면 stratified 분할(클래스 비율 유지)도 쉽게 할 수 있다.

**VLA 연결**: 로봇 데이터에서는 같은 에피소드의 프레임이 train과 test에 모두 들어가지 않도록 **에피소드 단위로 분할**하는 것이 중요하다. 이런 분할 전략을 Pandas의 `groupby`를 활용해 구현한다.

### 7. Pandas와 NumPy의 관계

Pandas는 내부적으로 NumPy 배열을 사용한다. 언제든 변환 가능하다.

```
array = df['width'].values           # Series → NumPy array
array = df[['width', 'height']].values  # DataFrame → 2D NumPy array

df = pd.DataFrame(array, columns=['width', 'height'])  # NumPy → DataFrame
```

일반적인 작업 흐름:
1. **Pandas**로 CSV를 읽고, 탐색하고, 전처리한다
2. **NumPy** 배열로 변환한다
3. **PyTorch/TensorFlow** 텐서로 변환해 모델에 입력한다

---

## 연습 주제

1. **CSV 읽기와 탐색**: 공개 데이터셋(예: Kaggle의 Titanic, Iris)의 CSV를 불러와서 `head()`, `shape`, `describe()`, `info()`, `value_counts()`를 사용해 데이터를 파악해 보자.

2. **필터링 연습**: 특정 조건(예: 나이 30 이상, 특정 클래스)에 해당하는 행만 추출해 보자. 두 가지 이상의 조건을 동시에 적용해 보자.

3. **결측치 탐색**: `isnull().sum()`으로 결측치를 확인하고, `dropna()`와 `fillna()`를 각각 적용해 보자. 결과의 행 수 변화를 비교해 보자.

4. **그룹별 통계**: `groupby`를 사용해 범주별 평균, 최댓값, 데이터 수를 구해 보자. 결과를 Matplotlib의 bar chart로 시각화해 보자.

5. **데이터 분할**: 임의의 DataFrame을 8:1:1 비율로 train/val/test로 나누고, 각 분할의 크기와 클래스 분포를 확인해 보자.

6. **NumPy 변환**: DataFrame의 수치형 열을 NumPy 배열로 변환하고, NumPy 연산(평균, 표준편차 등)을 적용한 후 다시 DataFrame으로 복원해 보자.

---

## 다음 단계

NumPy, Matplotlib, Pandas의 기초를 마쳤다. 이제 데이터를 다루는 도구를 갖추었으니, 본격적으로 **머신러닝의 핵심 개념**을 학습할 차례다.

다음은 [ml-basics/](../04-ml-basics/) 폴더로 이동하여 머신러닝 기초를 학습한다. 손실 함수, 경사 하강법, 과적합과 정규화 등 딥러닝의 근본 원리를 배우게 된다. 여기서 배운 NumPy 배열 연산, 시각화, 데이터 탐색 능력이 모든 단계에서 활용된다.
