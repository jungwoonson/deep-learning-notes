# Llama 아키텍처

## VLA와의 연결

**Llama 2 7B는 OpenVLA의 언어 모델 백본 그 자체이다.** 이 노트에서 다루는 모든 구성 요소(RMSNorm, SwiGLU, RoPE, GQA)가 OpenVLA 내부에서 그대로 작동한다. VLA가 로봇 행동을 생성할 때, 이미지 토큰과 언어 토큰이 바로 이 Llama 2 아키텍처를 통과하여 행동 토큰이 출력된다. 이 노트는 그 과정의 가장 구체적인 내부 구조를 설명한다.

---

## 핵심 개념

### 0. Llama 2 7B 전체 구조 개요

Llama 2 7B의 기본 구성:

| 항목 | 값 |
|------|-----|
| 레이어 수 | 32 |
| 히든 차원 (d_model) | 4,096 |
| Attention Head 수 | 32 |
| KV Head 수 (GQA) | 32 (Llama 2 7B는 MHA, 70B는 GQA 8) |
| FFN 히든 차원 | 11,008 |
| 어휘 크기 | 32,000 |
| 컨텍스트 길이 | 4,096 |
| 총 파라미터 | 약 6.7B |

각 Transformer 레이어의 구조:

```
입력 x
  -> RMSNorm -> Multi-Head Attention (with RoPE) -> + x (잔차 연결)
  -> RMSNorm -> SwiGLU FFN -> + x (잔차 연결)
출력
```

기존 Transformer와의 차이:
- LayerNorm -> **RMSNorm** (Pre-Norm 방식)
- ReLU/GELU FFN -> **SwiGLU** FFN
- 절대적 위치 임베딩 -> **RoPE** (상대적 회전 위치 임베딩)
- Multi-Head Attention -> **GQA** (70B에서)

이제 각 구성 요소를 하나씩 살펴보자.

### 1. RMSNorm (Root Mean Square Normalization)

#### 왜 정규화가 필요한가

딥러닝에서 레이어를 거칠수록 activation 값의 분포가 변동한다 (Internal Covariate Shift). 이를 안정화하기 위해 정규화가 필요하다.

#### LayerNorm vs RMSNorm

**LayerNorm (기존):**
- 평균(mean)을 빼고 표준편차(std)로 나눈다
- 두 가지 통계량(mean, std) 계산 필요

**RMSNorm (Llama가 사용):**
- 평균을 빼는 과정을 생략하고, RMS(Root Mean Square)로만 나눈다
- RMS(x) = sqrt(mean(x^2))
- 정규화: x / RMS(x) * gamma (gamma는 학습 가능한 스케일 파라미터)

**왜 RMSNorm인가?**
- 실험적으로 평균을 빼는 것(re-centering)은 성능에 큰 기여를 하지 않음
- 계산이 더 단순하여 **학습 속도가 빠르다** (약 10~15% 효율 향상)
- 성능은 LayerNorm과 동등하거나 더 나음

#### Pre-Norm vs Post-Norm

- **Post-Norm (원본 Transformer)**: Sublayer -> Add -> Norm
- **Pre-Norm (Llama)**: Norm -> Sublayer -> Add

Pre-Norm이 학습 안정성이 높다. 깊은 모델(32+ 레이어)에서 gradient가 더 잘 흐른다.

### 2. SwiGLU (Swish-Gated Linear Unit)

#### FFN의 역할 복습

Transformer 레이어에서 Attention은 "토큰 간 정보 교환"을 담당하고, FFN(Feed-Forward Network)은 "개별 토큰의 표현을 변환"하는 역할을 한다.

#### 기존 FFN

기존 Transformer FFN:
- x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)
- 두 개의 선형 변환 사이에 활성화 함수

#### SwiGLU의 구조

SwiGLU는 **세 개의 선형 변환**을 사용하는 gated 구조이다:

- gate = Swish(x * W_gate)
- up = x * W_up
- output = (gate * up) * W_down

여기서:
- **Swish 활성화 함수**: Swish(x) = x * sigmoid(x). ReLU보다 매끄럽고 음수 영역에서도 약간의 값을 통과시킴
- **Gating 메커니즘**: gate 값이 up 값을 요소별로 조절 (어떤 정보를 통과시킬지 결정)
- 파라미터 총량을 기존 FFN과 맞추기 위해 d_ff를 2/3로 줄임 (Llama 2: 11,008 = 4096 * 8/3 반올림)

**왜 SwiGLU인가?**
- PaLM 논문에서 ReLU, GELU, SwiGLU 등을 비교한 결과, SwiGLU가 가장 좋은 성능
- Gating이 정보 흐름을 더 세밀하게 조절
- Swish가 ReLU보다 부드러운 gradient를 제공

### 3. RoPE (Rotary Position Embedding)

#### 위치 정보가 왜 필요한가

Self-Attention은 본질적으로 **순서에 무관(permutation invariant)**하다. "고양이가 개를 쫓았다"와 "개가 고양이를 쫓았다"의 attention 값이 같아진다. 위치 정보를 별도로 주입해야 한다.

#### 위치 임베딩의 진화

1. **절대적 위치 임베딩 (원본 Transformer)**: 각 위치 0, 1, 2, ...에 고정된 벡터를 더함. 학습 가능 또는 sinusoidal
2. **상대적 위치 임베딩**: 두 토큰 사이의 "거리"를 인코딩. 위치 3과 5의 관계 = 위치 10과 12의 관계
3. **RoPE**: 상대적 위치 정보를 **회전(rotation)**으로 인코딩

#### RoPE의 핵심 아이디어

벡터를 2차원 평면에서의 회전으로 생각한다:

- 위치 m의 토큰 벡터를 m * theta 각도만큼 회전
- 위치 n의 토큰 벡터를 n * theta 각도만큼 회전
- Attention에서 두 벡터의 내적(dot product)을 계산하면, 결과는 **(m - n) * theta**에만 의존

이것이 의미하는 바:
- 내적 결과가 **상대적 거리 (m - n)**에만 의존한다
- 절대 위치가 아닌 상대 위치가 자연스럽게 인코딩된다
- 2차원을 넘어서, d_model 차원의 벡터를 d_model/2개의 2차원 평면으로 나누어 각각 다른 주파수로 회전

**구체적으로:**
- 벡터의 인접한 두 차원을 하나의 2차원 평면으로 묶음
- 각 평면마다 다른 주파수 theta_i를 사용 (저주파 ~ 고주파)
- 저주파: 먼 거리의 위치 관계를 인코딩
- 고주파: 가까운 거리의 위치 관계를 인코딩

**RoPE의 장점:**
- 학습 가능한 파라미터가 없다 (순수 수학적 변환)
- 상대적 위치 인코딩이 자연스럽게 달성된다
- 컨텍스트 길이 확장에 유연하다 (NTK-aware scaling, YaRN 등)

#### VLA에서의 의미

OpenVLA에서 이미지 토큰 + 언어 토큰 + 행동 토큰이 하나의 시퀀스를 이룬다. RoPE 덕분에 모델은 "이미지 토큰은 시퀀스 앞쪽에, 행동 토큰은 뒤쪽에" 있다는 위치 관계를 자연스럽게 파악한다.

### 4. GQA (Grouped-Query Attention)

#### 배경: MHA, MQA, GQA

**MHA (Multi-Head Attention, 기존):**
- Query, Key, Value 각각에 대해 H개의 head를 가진다
- 예: 32 heads -> Q 32개, K 32개, V 32개
- Llama 2 7B는 이 방식 사용

**MQA (Multi-Query Attention):**
- Query는 여전히 H개이지만, Key와 Value는 **1개만** 사용 (모든 head가 공유)
- 추론 시 KV 캐시가 H배 작아짐 -> 메모리 절약, 속도 향상
- 단점: 성능이 MHA보다 약간 떨어질 수 있음

**GQA (Grouped-Query Attention, Llama 2 70B):**
- MHA와 MQA의 중간 지점
- Query head들을 G개 그룹으로 나누고, 각 그룹이 하나의 KV head를 공유
- 예: 32 query heads, 8 KV heads -> 4개 query head가 1개 KV head를 공유

| 방식 | Query Heads | KV Heads | KV 캐시 크기 |
|------|-----------|----------|-------------|
| MHA | 32 | 32 | 100% |
| GQA (G=8) | 32 | 8 | 25% |
| MQA | 32 | 1 | 3.1% |

**왜 GQA인가?**
- MQA의 메모리 절약 효과를 대부분 유지하면서 성능 저하를 최소화
- Autoregressive 생성 시 KV 캐시가 병목이 되므로, KV 캐시 크기 감소 = 추론 속도 향상
- Llama 2 70B와 Llama 3 모든 모델에서 GQA 사용

#### KV Cache가 왜 중요한가

Autoregressive 생성 시 매 스텝마다 이전 토큰들의 Key, Value를 다시 계산하는 것은 낭비다. 한 번 계산한 KV를 저장(cache)해두면 새 토큰의 Q만 계산하면 된다. 시퀀스가 길어질수록 KV 캐시의 메모리 요구량이 커지며, GQA는 이를 직접적으로 줄인다.

---

## 연습 주제 (코드 없이 생각해보기)

1. **RMSNorm 수동 계산**: 벡터 `[3, 4]`의 RMS를 계산하고, RMSNorm을 적용한 결과를 구하라 (gamma = 1 가정). LayerNorm 결과와 비교하라.

2. **SwiGLU vs ReLU**: Swish(x) = x * sigmoid(x)에서 x = -1, 0, 1, 2 각각의 출력을 대략적으로 계산하고, ReLU와 비교하라. 음수 입력에서의 차이가 왜 중요할 수 있는가?

3. **RoPE 직관**: 위치 0의 벡터와 위치 5의 벡터의 attention score와, 위치 100의 벡터와 위치 105의 벡터의 attention score가 같다는 것이 왜 바람직한가? 절대적 위치 임베딩에서는 이것이 보장되는가?

4. **GQA 메모리 계산**: Llama 2 70B (GQA, 8 KV heads)의 시퀀스 길이 2048에서의 KV 캐시 크기를 MHA (64 KV heads) 방식과 비교하라. 몇 배의 메모리가 절약되는가?

5. **아키텍처 통합 이해**: "이미지: 테이블 위의 빨간 컵"이라는 입력이 Llama 2 레이어 하나를 통과하는 과정을 RMSNorm -> Attention(RoPE) -> RMSNorm -> SwiGLU 순서로 설명하라. 각 단계에서 어떤 처리가 이루어지는가?

6. **VLA 연결**: OpenVLA에서 이미지 토큰이 256개, 언어 토큰이 20개, 행동 토큰이 7개일 때 총 시퀀스 길이는 283이다. 이 시퀀스가 32개 Transformer 레이어를 통과할 때 KV 캐시의 크기가 왜 중요한가?

---
