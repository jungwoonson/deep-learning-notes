# 배열과 연산 (NumPy Arrays and Operations)

## 왜 필요한가: VLA로 가는 길

VLA(Vision-Language-Action) 모델은 이미지, 텍스트, 로봇 동작 데이터를 모두 **숫자 배열(array)** 로 표현한다.

- **Vision**: 카메라 이미지 한 장 = `(높이, 너비, 채널)` 형태의 3차원 배열
- **Language**: 토큰 임베딩 시퀀스 = `(시퀀스 길이, 임베딩 차원)` 형태의 2차원 배열
- **Action**: 로봇 관절 각도 = `(타임스텝, 관절 수)` 형태의 2차원 배열

딥러닝의 모든 연산은 결국 **다차원 배열의 생성, 변형, 연산**이다. NumPy는 이 모든 것의 기초 도구다.

---

## 핵심 개념

### 1. ndarray 생성 (Creation)

NumPy의 핵심 자료구조는 **ndarray**(n-dimensional array)다. Python 리스트와 달리 모든 원소가 같은 타입이고, 연산이 훨씬 빠르다.

```
np.array([1, 2, 3])           # 리스트로부터 생성
np.zeros((3, 4))              # 3x4 영행렬
np.ones((2, 3))               # 2x3 일행렬
np.random.randn(3, 4)         # 3x4 표준정규분포 난수
np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)          # 0~1 사이 균등 5개
```

**VLA 연결**: 신경망의 가중치(weight)를 초기화할 때 `randn`처럼 난수로 배열을 만드는 것이 첫 단계다.

### 2. dtype (데이터 타입)

배열의 각 원소가 어떤 타입인지를 dtype이 결정한다.

| dtype | 설명 | 딥러닝에서의 용도 |
|-------|------|-------------------|
| `float32` | 32비트 부동소수점 | 기본 학습 정밀도 |
| `float16` | 16비트 부동소수점 | 메모리 절약 (mixed precision) |
| `int64` | 64비트 정수 | 토큰 인덱스, 레이블 |
| `bool` | 불리언 | 마스크 (attention mask 등) |

**VLA 연결**: Transformer 모델에서 attention mask는 `bool` 배열이고, 토큰 ID는 `int64`, 임베딩은 `float32`다. dtype을 잘못 쓰면 메모리가 낭비되거나 연산이 실패한다.

### 3. shape, reshape, transpose

**shape**는 배열의 구조를 나타내는 튜플이다.

```
a = np.zeros((2, 3, 4))
a.shape                     # (2, 3, 4)
a.ndim                      # 3 (차원 수)
a.size                      # 24 (전체 원소 수)
```

**reshape**는 원소 수를 유지하면서 형태를 변경한다.

```
a = np.arange(12)            # shape: (12,)
b = a.reshape(3, 4)          # shape: (3, 4)
c = a.reshape(2, 2, 3)       # shape: (2, 2, 3)
d = a.reshape(3, -1)         # -1은 자동 계산 → (3, 4)
```

**transpose**는 축의 순서를 바꾼다.

```
a = np.zeros((2, 3))
a.T                          # shape: (3, 2)
a.transpose(1, 0)            # 위와 동일

# 3차원 이상
img = np.zeros((H, W, C))
img.transpose(2, 0, 1)       # (C, H, W) — PyTorch 이미지 형식
```

**VLA 연결**: 이미지 데이터는 라이브러리마다 축 순서가 다르다. OpenCV는 `(H, W, C)`, PyTorch는 `(C, H, W)`. transpose로 변환하는 것은 실무에서 매일 하는 일이다.

### 4. 인덱싱 (Indexing)

#### Basic Indexing (기본 인덱싱)

```
a = np.array([[1, 2, 3],
              [4, 5, 6]])
a[0, 1]                     # 2 (0행 1열)
a[1, :]                     # [4, 5, 6] (1행 전체)
a[:, 2]                     # [3, 6] (2열 전체)
a[0:1, 1:3]                 # [[2, 3]] (슬라이싱)
```

#### Fancy Indexing (팬시 인덱싱)

정수 배열로 원하는 위치를 직접 지정한다.

```
a = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 3, 4])
a[idx]                       # [10, 40, 50]
```

**VLA 연결**: 토큰 임베딩 테이블에서 토큰 ID로 해당 벡터를 꺼내는 것이 바로 fancy indexing이다. `embedding_table[token_ids]`처럼 쓴다.

#### Boolean Indexing (불리언 인덱싱)

조건을 만족하는 원소만 선택한다.

```
a = np.array([1, -2, 3, -4, 5])
a[a > 0]                     # [1, 3, 5]
```

**VLA 연결**: attention mask에서 패딩이 아닌 실제 토큰만 골라내는 것이 boolean indexing이다.

### 5. Element-wise 연산 (원소별 연산)

배열 간 연산은 같은 위치의 원소끼리 수행된다.

```
a + b        # 원소별 덧셈
a * b        # 원소별 곱셈 (행렬 곱이 아님!)
a / b        # 원소별 나눗셈
np.exp(a)    # 원소별 지수함수
np.log(a)    # 원소별 로그
np.maximum(a, 0)  # 원소별 max → ReLU 활성화 함수의 원리
```

**VLA 연결**: 활성화 함수(ReLU, sigmoid 등)는 모두 element-wise 연산이다. `np.maximum(a, 0)`이 ReLU의 전부다.

### 6. Axis 기반 연산 (축 연산)

`axis` 파라미터는 "어느 축을 따라 줄일 것인가"를 지정한다.

```
a = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape: (2, 3)

np.sum(a)              # 21 (전체 합)
np.sum(a, axis=0)      # [5, 7, 9] (행 방향으로 합 → 결과 shape: (3,))
np.sum(a, axis=1)      # [6, 15] (열 방향으로 합 → 결과 shape: (2,))

np.mean(a, axis=1)     # [2.0, 5.0]
np.max(a, axis=0)      # [4, 5, 6]
np.argmax(a, axis=1)   # [2, 2] (각 행에서 최댓값의 인덱스)
```

축 연산의 핵심 규칙: **지정한 axis가 사라진다**.
- `(2, 3)`에서 `axis=0` → shape `(3,)` (0번 축 사라짐)
- `(2, 3)`에서 `axis=1` → shape `(2,)` (1번 축 사라짐)

`keepdims=True`를 쓰면 사라진 축이 크기 1로 유지된다:
- `(2, 3)`에서 `axis=1, keepdims=True` → shape `(2, 1)`

**VLA 연결**:
- `softmax`는 특정 axis를 따라 `exp` + `sum`을 적용한다. attention score 계산의 핵심이다.
- `argmax(axis=-1)`는 모델의 예측 클래스를 구할 때 사용한다.
- `mean(axis=0)`는 배치 단위 평균을 구할 때 (batch normalization 등) 사용한다.

---

## 연습 주제

아래 주제들을 직접 코드로 실험해 보자. 정답 코드를 외우는 것이 아니라 **동작 원리를 이해하는 것**이 목표다.

1. **배열 생성 연습**: 다양한 방법(`zeros`, `ones`, `arange`, `linspace`, `random`)으로 배열을 만들고 shape과 dtype을 확인해 보자.

2. **reshape 실험**: `(24,)` 배열을 `(2,3,4)`, `(4,6)`, `(6,4)` 등으로 변환해 보고, 불가능한 reshape(`(5,5)` 등)을 시도하면 어떤 에러가 나는지 확인해 보자.

3. **이미지 축 변환**: `(H, W, C)` 형태의 가짜 이미지 배열을 만들고, `transpose`로 `(C, H, W)` 형태로 바꿔 보자.

4. **인덱싱 종합 연습**: 2차원 배열에서 basic, fancy, boolean 인덱싱을 각각 사용해 원하는 부분을 추출해 보자.

5. **axis 이해**: `(3, 4)` 배열에 대해 `sum`, `mean`, `max`, `argmax`를 `axis=0`, `axis=1`, `axis=None`으로 각각 적용하고, 결과의 shape을 예측한 후 확인해 보자.

6. **ReLU 구현**: `np.maximum`을 이용해 음수를 0으로 바꾸는 ReLU 함수를 만들어 보자.

---

## 다음 노트

다음은 [브로드캐스팅과 벡터화 (Broadcasting and Vectorization)](./02-broadcasting-vectorization.md)를 학습한다. NumPy가 서로 다른 shape의 배열을 어떻게 자동으로 맞춰서 연산하는지, 그리고 왜 for 루프 대신 벡터 연산을 써야 하는지를 배운다.
