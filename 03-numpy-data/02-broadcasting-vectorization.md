# 브로드캐스팅과 벡터화 (Broadcasting and Vectorization)

## 왜 필요한가: VLA로 가는 길

VLA 모델의 연산은 항상 **서로 다른 shape의 텐서가 만나는 상황**이다.

- 이미지 배치 `(B, C, H, W)`에 채널별 정규화 값 `(C, 1, 1)`을 빼야 한다
- 임베딩 벡터 `(seq_len, d_model)`에 스칼라 `1/sqrt(d_model)`을 곱해야 한다
- Attention score `(B, heads, seq, seq)`에 마스크 `(1, 1, 1, seq)`를 더해야 한다

이 모든 상황에서 shape이 안 맞는데도 연산이 되는 이유가 **broadcasting**이다. 그리고 이 연산을 for 루프 없이 한 번에 처리하는 것이 **vectorization**이다. 이 두 개념을 모르면 딥러닝 코드를 읽을 수도, 쓸 수도 없다.

---

## 핵심 개념

### 1. Broadcasting 규칙

Broadcasting은 shape이 다른 두 배열을 연산할 때 NumPy가 자동으로 shape을 맞추는 메커니즘이다. 실제로 메모리를 복사하지 않으므로 효율적이다.

**3가지 규칙** (오른쪽 축부터 비교):

1. **차원 수가 다르면**, 적은 쪽의 왼쪽에 크기 1인 축을 추가한다
2. **같은 위치의 축 크기가 다르면**, 크기 1인 쪽을 다른 쪽 크기로 늘린다
3. **크기가 1도 아니고 같지도 않으면**, 에러가 발생한다

#### 예시로 이해하기

```
(4, 3) + (3,)       → (4, 3) + (1, 3) → (4, 3) + (4, 3) ✓
(4, 3) + (4, 1)     → (4, 3) + (4, 3)                     ✓
(4, 3) + (4,)       → (4, 3) + (1, 4) → 에러! 3 ≠ 4      ✗
(3, 1) + (1, 4)     → (3, 4) + (3, 4)                     ✓
```

#### 자주 등장하는 패턴

| 패턴 | A shape | B shape | 결과 shape | 딥러닝 용도 |
|------|---------|---------|-----------|------------|
| 스칼라 연산 | `(M, N)` | `()` | `(M, N)` | learning rate 곱하기 |
| 행 빼기 | `(M, N)` | `(N,)` | `(M, N)` | 평균 빼기 (정규화) |
| 열 빼기 | `(M, N)` | `(M, 1)` | `(M, N)` | 샘플별 바이어스 |
| 외적 형태 | `(M, 1)` | `(1, N)` | `(M, N)` | attention score 계산 |

#### 핵심을 관통하는 하나의 원리

Broadcasting은 결국 **"크기 1인 축은 필요한 만큼 반복될 수 있다"** 는 원리다. 크기 1이 아닌 축은 절대 변하지 않는다.

### 2. Vectorization (벡터화)

벡터화란 **for 루프를 배열 연산 하나로 대체하는 것**이다.

#### 루프 vs 벡터화

개념적 비교:

```
# 루프 방식 (느림)
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])

# 벡터화 방식 (빠름)
result = a + b
```

두 코드는 같은 결과를 내지만, 벡터화 버전이 **수십~수백 배 빠르다**. 이유는:
- NumPy 내부가 C/Fortran으로 구현되어 있다
- CPU의 SIMD 명령어를 활용한다
- Python 인터프리터의 오버헤드가 없다

#### 실전에서 벡터화가 적용되는 곳

| 연산 | 루프적 사고 | 벡터화 사고 |
|------|-----------|-----------|
| 유클리드 거리 | 각 원소 차이를 구하고 제곱해서 합산 | `np.sqrt(np.sum((a - b)**2))` |
| 행렬 곱 | 이중 루프로 내적 계산 | `a @ b` |
| Softmax | 각 원소에 exp 적용 후 합으로 나누기 | `np.exp(x) / np.sum(np.exp(x))` |
| Batch 정규화 | 각 샘플에 대해 평균/분산 계산 | `(x - x.mean(axis=0)) / x.std(axis=0)` |

### 3. "행렬로 생각하라, 루프로 생각하지 마라"

딥러닝 코드를 잘 작성하려면 사고방식 자체를 바꿔야 한다.

#### 루프적 사고 (피해야 할 것)

> "각 데이터 포인트에 대해, 각 가중치를 곱하고, 하나씩 더해서..."

#### 행렬적 사고 (목표)

> "입력 행렬과 가중치 행렬을 곱하면 출력 행렬이 나온다"

```
# 루프적 사고: 뉴런 하나의 출력
output = 0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]
output += bias

# 행렬적 사고: 전체 레이어의 출력
output = inputs @ weights + bias
```

**VLA 연결**: Transformer의 self-attention을 수식으로 쓰면 다음과 같다.

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

이 한 줄에 broadcasting(`/ sqrt(d_k)`), 행렬 곱(`@`), 벡터화된 softmax가 모두 들어 있다. 루프로 생각하면 이 수식을 이해할 수 없다.

### 4. 흔한 실수와 디버깅

#### shape 불일치 에러

```
ValueError: operands could not be broadcast together with shapes (3,4) (3,)
```

이 에러가 나면: 오른쪽 축부터 비교해 보자. `(3,4)`와 `(3,)` → 오른쪽 축이 4와 3으로 불일치.

해결: `(3,)` → `(3,1)`로 reshape하면 broadcasting 가능.

```
b = b.reshape(-1, 1)    # (3,) → (3, 1)
# 또는
b = b[:, np.newaxis]     # 같은 효과
```

#### np.newaxis 활용

`np.newaxis`(= `None`)는 크기 1인 축을 삽입한다. Broadcasting을 맞추기 위해 자주 쓴다.

```
a = np.array([1, 2, 3])     # shape: (3,)
a[:, np.newaxis]             # shape: (3, 1) — 열벡터
a[np.newaxis, :]             # shape: (1, 3) — 행벡터
```

---

## 연습 주제

1. **Broadcasting 규칙 손풀기**: 아래 shape 조합의 broadcasting 결과를 종이에 먼저 예측하고, 코드로 확인해 보자.
   - `(5, 3)` + `(3,)`
   - `(5, 3)` + `(5, 1)`
   - `(4, 1, 3)` + `(1, 5, 3)`
   - `(2, 3)` + `(4,)` (에러 발생하는지 확인)

2. **속도 비교**: 크기 100만인 배열 두 개의 원소별 곱을 for 루프와 `a * b`로 각각 수행하고, 시간을 비교해 보자. (`time` 모듈이나 `%timeit` 사용)

3. **Softmax 벡터화**: 2차원 배열에서 각 행에 대해 softmax를 for 루프 없이 계산해 보자. (`np.exp`, `np.sum`, `axis`, `keepdims` 활용)

4. **이미지 정규화**: `(H, W, 3)` 형태의 가짜 이미지에서 채널별 평균 `(3,)`을 빼는 정규화를 broadcasting으로 수행해 보자.

5. **유클리드 거리 행렬**: N개의 점 `(N, D)`에 대해 모든 쌍의 거리를 루프 없이 `(N, N)` 행렬로 계산해 보자. (힌트: `np.newaxis`로 `(N, 1, D)`와 `(1, N, D)`를 만들어 broadcasting)

---

## 다음 노트

다음은 [Matplotlib 시각화 (Matplotlib Visualization)](./03-matplotlib-visualization.md)를 학습한다. 배열을 눈으로 확인하는 방법을 배운다. 이미지 시각화, 학습 곡선 그리기, attention heatmap 표현 등 딥러닝에서 필수적인 시각화 기법을 다룬다.
