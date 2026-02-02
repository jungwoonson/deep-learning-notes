# 옵티마이저 (Optimizers: SGD, Adam)

## 왜 알아야 하는가 (Why This Matters for VLA)

옵티마이저(Optimizer)는 역전파로 구한 gradient를 사용하여 **파라미터를 실제로 업데이트하는 전략**이다. 같은 네트워크, 같은 데이터라도 옵티마이저에 따라 학습 속도와 최종 성능이 크게 달라진다.

VLA와의 연결 고리:
- VLA를 포함한 거의 모든 Transformer 기반 모델은 **AdamW**를 사용한다
  - Adam + Weight Decay의 올바른 결합
  - GPT, Llama, ViT, OpenVLA, pi-zero 모두 AdamW 사용
- **학습률 스케줄링**은 대규모 모델 학습의 필수 요소이다
  - Warmup + Cosine Decay가 Transformer/VLA의 사실상 표준
  - 학습률이 잘못되면 수백 GPU-hour를 낭비할 수 있다
- 파인튜닝(fine-tuning) 시 옵티마이저 설정이 사전학습과 달라야 하는 경우가 많다
  - VLA 파인튜닝: 더 낮은 학습률, 짧은 warmup

---

## 핵심 개념 (Core Concepts)

### 1. SGD (Stochastic Gradient Descent)

가장 기본적인 옵티마이저. Gradient의 반대 방향으로 파라미터를 업데이트한다.

```
업데이트 규칙:
  w = w - lr * gradient

lr: 학습률 (learning rate)
gradient: dL/dw (역전파로 구한 기울기)
```

**비유**:
```
산에서 내려가기:
  현재 위치에서 가장 가파른 방향(gradient)을 찾고,
  그 반대 방향으로 한 걸음(lr) 내딛는다.

문제점:
  - 계곡이 좁고 길면 좌우로 진동하며 느리게 전진
  - 지역 최소점(local minimum)에 갇힐 수 있음
  - 안장점(saddle point)에서 멈출 수 있음

손실 곡면:
     \   /  ← 좁은 계곡에서 SGD는
      \ /      좌우로 진동하며 느리게 이동
       V
       ↓ (목표: 아래로 빨리 가고 싶음)
```

### 2. Momentum (모멘텀)

SGD에 **관성(momentum)**을 추가하여 진동을 줄이고 가속한다.

```
업데이트 규칙:
  v = beta * v + gradient              (속도 업데이트)
  w = w - lr * v                       (파라미터 업데이트)

beta: 모멘텀 계수 (보통 0.9)
v: 속도 (velocity) - 이전 gradient들의 가중 평균

초기값: v = 0
```

**직관적 이해**:
```
공이 언덕을 굴러 내려가는 것:
  - 이전에 움직이던 방향으로 관성이 작용
  - 같은 방향의 gradient가 계속되면 속도가 누적 → 가속
  - 반대 방향의 gradient가 오면 속도가 감소 → 진동 억제

SGD:                    Momentum:
     ↗↙                     ↘
    ↗↙                       ↘
   ↗↙  (좌우 진동)              ↘  (관성으로 직진)
  ↗↙                             ↘
   ↓                               ↓ (빠르게 수렴)
```

**beta=0.9의 의미**: 현재 gradient보다 **과거 gradient의 축적**이 9배 더 영향력이 있다. 방향이 일관되면 가속하고, 진동하면 상쇄한다.

### 3. RMSProp (Root Mean Square Propagation)

각 파라미터별로 **학습률을 자동 조정**한다. Gradient가 큰 파라미터는 작게, 작은 파라미터는 크게 업데이트한다.

```
업데이트 규칙:
  s = beta * s + (1 - beta) * gradient^2    (gradient 제곱의 이동 평균)
  w = w - lr * gradient / (sqrt(s) + epsilon)

beta: 감쇠 계수 (보통 0.999)
epsilon: 0으로 나누기 방지 (보통 1e-8)
s: gradient 제곱의 지수이동평균 (EMA)
```

**핵심 아이디어: 적응적 학습률 (Adaptive Learning Rate)**
```
gradient가 크고 자주 변하는 파라미터:
  s가 큼 → sqrt(s)가 큼 → 나누면 작은 업데이트
  "이미 많이 변하고 있으니 조금씩 조정"

gradient가 작고 드문 파라미터:
  s가 작음 → sqrt(s)가 작음 → 나누면 큰 업데이트
  "충분히 변하지 않았으니 더 크게 조정"

→ 각 파라미터가 자신만의 적응적 학습률을 갖게 된다
```

### 4. Adam (Adaptive Moment Estimation)

**Momentum + RMSProp**을 결합한 옵티마이저. 현대 딥러닝에서 가장 널리 사용된다.

```
업데이트 규칙:
  m = beta1 * m + (1 - beta1) * gradient          (1차 모멘트: gradient의 평균)
  v = beta2 * v + (1 - beta2) * gradient^2         (2차 모멘트: gradient 제곱의 평균)

  편향 보정 (Bias Correction):
  m_hat = m / (1 - beta1^t)                        (t는 현재 step 번호)
  v_hat = v / (1 - beta2^t)

  w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

기본 하이퍼파라미터:
  beta1 = 0.9     (Momentum과 같은 역할)
  beta2 = 0.999   (RMSProp과 같은 역할)
  epsilon = 1e-8
  lr = 0.001
```

**편향 보정이 필요한 이유**:
```
m과 v를 0으로 초기화하면, 초기 step에서 이들이 0에 편향됨.

예: step 1에서 gradient = 5.0, beta1 = 0.9
  m = 0.9 * 0 + 0.1 * 5.0 = 0.5  (실제 gradient는 5.0인데 0.5로 추정)

편향 보정:
  m_hat = 0.5 / (1 - 0.9^1) = 0.5 / 0.1 = 5.0  (올바른 추정!)

t가 커질수록 (1 - beta^t)이 1에 가까워져 보정 효과가 사라짐 → 자연스럽게 해소
```

**Adam이 왜 강력한가**:
```
Momentum의 장점: 방향 정보 활용, 진동 감소, 가속
RMSProp의 장점: 파라미터별 적응적 학습률

Adam = 둘의 결합
  → 방향이 일관되면 가속 (from Momentum)
  → 파라미터별로 적절한 크기로 업데이트 (from RMSProp)
  → 대부분의 문제에서 빠르고 안정적으로 수렴
```

### 5. AdamW (Adam with Decoupled Weight Decay)

Adam의 **weight decay 구현을 올바르게 수정**한 버전. Transformer/VLA의 사실상 표준.

```
Adam에서의 L2 정규화 (문제):
  gradient_regularized = gradient + lambda * w
  이 regularized gradient를 Adam에 넣으면:
  → 적응적 학습률이 정규화 항에도 적용됨
  → weight decay의 효과가 왜곡됨

AdamW (해결):
  1. gradient를 Adam으로 업데이트 (정규화 없이)
  2. weight decay를 별도로 적용
  w = w - lr * (adam_update + lambda * w)

이것이 "decoupled weight decay"이다.
```

**실무에서의 차이**:
```
Adam + L2:    정규화가 적응적 학습률에 의해 왜곡
AdamW:        정규화가 독립적으로 작용 → 일관된 효과

→ Transformer 학습에서 AdamW가 Adam보다 일관적으로 더 좋은 성능
→ GPT-3, Llama 2, ViT, OpenVLA 등 모두 AdamW 사용

일반적인 AdamW 설정 (Transformer/VLA):
  lr:         1e-4 ~ 3e-4 (사전학습), 1e-5 ~ 5e-5 (파인튜닝)
  beta1:      0.9
  beta2:      0.95 또는 0.999
  epsilon:    1e-8
  weight_decay: 0.01 ~ 0.1
```

### 6. 옵티마이저 비교 총정리

```
옵티마이저       핵심 아이디어              장점                    단점
SGD             gradient 반대 방향        단순함, 일반화 좋음      느림, 진동
SGD+Momentum    관성 추가                가속, 진동 감소          lr 민감
RMSProp         적응적 학습률             파라미터별 최적화        불안정할 수 있음
Adam            Momentum + RMSProp       빠르고 안정적            메모리 2배 (m, v)
AdamW           Adam + 올바른 정규화      Transformer 최적        메모리 2배 (m, v)

메모리 비교 (파라미터 수 N):
  SGD:           N         (파라미터만)
  SGD+Momentum:  2N        (파라미터 + 속도)
  Adam/AdamW:    3N        (파라미터 + m + v)

VLA (7B 파라미터) 기준:
  SGD:    ~28GB
  Adam:   ~84GB → GPU 메모리의 큰 부분을 차지
```

### 7. 학습률 스케줄링 (Learning Rate Scheduling)

학습 내내 같은 학습률을 사용하는 것은 비효율적이다. **학습 단계에 따라 학습률을 변화**시키는 것이 학습률 스케줄링이다.

#### Warmup (워밍업)

```
학습 초기에 학습률을 0에서 서서히 올리는 기법.

lr
 |    /← 목표 학습률
 |   /
 |  /
 | /
 |/
 +-------- step
  warmup

왜 필요한가:
  학습 초기에는 파라미터가 랜덤 → gradient 방향이 불안정
  갑자기 큰 학습률을 적용하면 학습이 발산할 수 있음
  → 처음에는 조심스럽게, 점차 가속

Transformer/VLA에서 필수적인 이유:
  - Adam의 m, v가 초기에 0으로 시작 → 편향 보정이 있어도 불안정
  - LayerNorm/RMSNorm의 통계가 초기에 부정확
  - 일반적으로 전체 학습의 1~5%를 warmup에 사용
    예: 100,000 step 학습 → 1,000~5,000 step warmup
```

#### Cosine Annealing (코사인 감쇠)

```
코사인 함수를 따라 학습률을 점진적으로 감소시킴.

lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

lr
 |\.
 |  '-.
 |     '-.
 |        '._
 |           '-._ lr_min
 +------------------→ step
  0                T

특징:
  - 초반에는 천천히 감소, 중반에 빠르게, 후반에 다시 천천히
  - Step Decay보다 부드럽고 성능이 더 좋은 경향
  - Transformer/VLA 학습에서 가장 인기 있는 스케줄
```

#### Warmup + Cosine Decay (표준 스케줄)

```
Transformer/VLA의 사실상 표준 학습률 스케줄:

lr
 |    /\
 |   /  '-.
 |  /      '-.
 | /          '._
 |/               '-._ lr_min
 +-----|------------------→ step
   warmup    cosine decay

구체적 예시 (Llama 2 학습):
  - Peak lr: 3e-4
  - Warmup: 2,000 steps
  - Total: 2,000,000 steps
  - Cosine decay to lr_min = 3e-5 (peak의 1/10)

VLA 파인튜닝 예시:
  - Peak lr: 2e-5
  - Warmup: 500 steps
  - Total: 50,000 steps
  - Cosine decay to 0
```

#### Linear Warmup + Linear Decay

```
BERT 등 일부 모델에서 사용:

lr
 |    /\
 |   /  \
 |  /    \
 | /      \
 |/        \
 +-----|----\---------→ step
   warmup    linear decay

cosine보다 단순하지만 효과적
```

### 8. 학습률 진단 (Learning Rate Diagnosis)

```
학습률이 잘 설정되었는지 loss 곡선으로 확인:

적절한 lr:           lr 너무 높음:         lr 너무 낮음:
loss                  loss                  loss
 |\.                  | /\  /\             |\
 |  '-.               |/  \/  \            | \
 |     '-.            |        ...         |  '...............
 |        '-._        |                    |
 +--------step       +--------step        +--------step
(부드럽게 감소)      (진동 또는 발산)      (매우 느리게 감소)

warmup이 부족할 때:                warmup이 적절할 때:
loss                               loss
 |  /\                             |\.
 | /  ↘                            |  '-.
 |/    '-.                         |     '-.
 +--------step                    +--------step
(초반에 치솟았다 내려옴)           (부드럽게 시작)
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **SGD vs Momentum 비교**: 좁고 긴 계곡 모양의 손실 곡면에서 SGD가 진동하는 이유를 설명하라. Momentum이 이를 어떻게 해결하는지 beta=0.9 기준으로 구체적으로 설명하라.

2. **Adam 수식 이해**: beta1=0.9, beta2=0.999일 때, step 1에서 편향 보정 전후의 m과 v가 얼마나 달라지는지 계산하라. step 100에서는?

3. **메모리 계산**: 70억 파라미터 모델을 float32로 학습할 때, SGD와 AdamW 각각에 필요한 옵티마이저 상태(optimizer state) 메모리를 계산하라.

4. **학습률 스케줄 설계**: 총 100,000 step을 학습하는 VLA 모델에 대해, warmup 2,000 step + cosine decay 스케줄을 설계하라. peak lr = 2e-4, min lr = 2e-5일 때 step 0, 1000, 2000, 50000, 100000에서의 학습률을 대략적으로 계산하라.

5. **AdamW vs Adam**: 왜 AdamW가 Adam + L2보다 Transformer에서 더 좋은 성능을 보이는지, "적응적 학습률이 weight decay를 왜곡한다"는 것이 구체적으로 무엇을 의미하는지 설명하라.

6. **실전 디버깅**: loss가 처음 몇 step에서 급격히 증가한 후 감소하기 시작한다. 이 현상의 원인은 무엇이고, 어떻게 해결하겠는가?

---

## 다음 노트 (Next Note)

옵티마이저로 학습 전략을 세웠다. 하지만 깊은 네트워크를 안정적으로 학습시키려면 **정규화(normalization)**와 **과적합 방지(dropout)** 기법이 필요하다. 특히 Transformer에서 사용되는 LayerNorm과 RMSNorm을 이해해야 한다.

**다음**: [정규화와 드롭아웃 (Normalization and Dropout)](./05-normalization-dropout.md) - BatchNorm, LayerNorm, RMSNorm과 Dropout, 가중치 초기화를 다룬다.
