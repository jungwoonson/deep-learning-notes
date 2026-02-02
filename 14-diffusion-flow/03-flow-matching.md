# Flow Matching

## VLA 연결고리

Flow matching은 2025~2026년 VLA에서 **가장 주목받는 action 생성 방법**이다. pi-0 (Physical Intelligence), SmolVLA (HuggingFace), GR00T N1 (NVIDIA) 등 최신 VLA 시스템이 모두 flow matching을 채택했다. Diffusion보다 단순하고 빠르며, 연속 행동 생성에 특히 적합하다.

---

## 핵심 개념

### 1. Diffusion의 한계와 Flow Matching의 동기

**Diffusion의 문제점**:
- Forward process가 복잡한 확률 과정(stochastic process)이다
- 수백~수천 step의 반복적 denoising이 필요하다
- 학습 목표의 유도가 수학적으로 복잡하다
- Noise schedule 설계에 민감하다

**Flow matching의 제안**: "노이즈에서 데이터로 가는 가장 간단한 경로를 학습하면 어떨까?"

### 2. Flow Matching의 핵심 아이디어

노이즈 분포에서 데이터 분포로의 **연속적인 변환(flow)**을 학습한다.

**핵심 구성 요소**:
- 시작점: 노이즈 $z \sim \mathcal{N}(0, I)$ ($t=0$)
- 끝점: 데이터 $x_1$ ($t=1$)
- 경로: $t=0$에서 $t=1$까지의 연속적인 변환
- Network: 각 시점 $t$에서의 **velocity**(속도 벡터)를 예측한다

$$\frac{dx}{dt} = v_\theta(x_t, t)$$

Network $v_\theta$가 "현재 위치 $x_t$와 시각 $t$에서 어느 방향으로 얼마나 빠르게 이동해야 하는지"를 알려준다.

### 3. Straight-Line Interpolation (직선 보간)

Flow matching에서 가장 간단하고 효과적인 경로 설정이다.

**아이디어**: 노이즈 $z$와 데이터 $x_1$ 사이를 직선으로 연결한다.

$$x_t = (1 - t) \cdot z + t \cdot x_1$$

- $t=0$일 때: $x_0 = z$ (순수 노이즈)
- $t=1$일 때: $x_1 = x_1$ (데이터)
- $t=0.5$일 때: 노이즈와 데이터의 정확히 중간

**목표 velocity**: 이 직선 경로를 따르는 속도는 상수이다.

$$v_{\text{target}} = x_1 - z$$

**학습**: Network가 이 목표 velocity를 예측하도록 학습한다.

$$\mathcal{L} = \| v_\theta(x_t, t) - (x_1 - z) \|^2$$

놀랍도록 단순하다. 이것이 flow matching의 핵심이다.

### 4. Optimal Transport (최적 수송)

직선 보간을 더 발전시킨 개념이다.

**문제**: 노이즈 점 $z$와 데이터 점 $x_1$을 어떻게 짝짓는가?

**Optimal Transport 관점**: 노이즈 분포를 데이터 분포로 변환할 때, 전체적인 "이동 비용"을 최소화하는 매핑을 찾는다.

**간단한 경우** (Conditional Flow Matching):
- 학습 데이터 $x_1$ 하나와 랜덤 노이즈 $z$ 하나를 짝짓는다
- 둘 사이의 직선 경로를 따르는 velocity를 학습한다
- 이것만으로도 전체 분포 수준의 flow를 잘 학습한다

**결과**: 경로가 서로 교차하지 않는 "깔끔한" flow를 얻을 수 있다. 교차가 적을수록 더 적은 step으로 정확한 생성이 가능하다.

### 5. Diffusion vs Flow Matching 비교

| 측면 | Diffusion (DDPM) | Flow Matching |
|------|-------------------|---------------|
| **경로** | 확률적 (stochastic), 복잡 | 결정적 (deterministic), 직선 |
| **학습 목표** | 노이즈 $\epsilon$ 예측 | 속도 벡터 $v$ 예측 |
| **수학적 복잡도** | SDE, Markov chain 이론 필요 | ODE, 직선 보간만으로 충분 |
| **Noise schedule** | 중요, 설계 필요 | 불필요 (직선이므로) |
| **생성 step 수** | 수십~수백 step | 더 적은 step으로 가능 |
| **생성 품질** | 우수 | 동등 또는 우수 |
| **구현 난이도** | 보통 | 더 간단 |

### 6. Flow Matching의 장점 (VLA 관점)

| 장점 | 설명 |
|------|------|
| **빠른 추론** | 적은 step으로 생성 -> 로봇 제어 주파수에 유리 |
| **단순한 학습** | Velocity 예측이라는 직관적 목표 |
| **연속 행동에 적합** | Multimodal action distribution을 자연스럽게 표현 |
| **안정적 학습** | Noise schedule 없이 안정적으로 수렴 |
| **Action chunk 생성** | 여러 time step의 action을 한 번에 생성 가능 |

### 7. Flow Matching을 채택한 주요 VLA 시스템

| 시스템 | 개발사 | 연도 | Flow matching 역할 |
|--------|--------|------|-------------------|
| **pi-0** | Physical Intelligence | 2024 | VLM + flow matching action head |
| **SmolVLA** | HuggingFace | 2025 | 경량 VLA, flow matching decoder |
| **GR00T N1** | NVIDIA | 2025 | 휴머노이드 로봇 제어 |
| **pi-0.5** | Physical Intelligence | 2025 | pi-0의 후속, 더 다양한 로봇 |

이들의 공통점: VLM(Vision-Language Model)이 시각-언어 이해를 담당하고, flow matching head가 연속 행동을 생성한다. Diffusion 대비 추론이 빠르고 학습이 간단하여 대규모 시스템에 적합하다.

---

## 연습 주제 (코드 없이)

1. 직선 보간 x_t = (1-t)*z + t*x_1에서 t=0, 0.25, 0.5, 0.75, 1.0 각각에서 x_t가 어떤 값인지 z=0, x_1=10으로 계산하라.
2. Flow matching의 목표 velocity v = x_1 - z가 시간 t에 의존하지 않는 이유를 설명하라. (힌트: 직선의 기울기)
3. Diffusion에서 noise schedule을 잘못 설정하면 생기는 문제가, flow matching에서는 왜 발생하지 않는지 설명하라.
4. 로봇이 10Hz로 제어되고, diffusion은 50 step, flow matching은 10 step이 필요하다면, 각각 행동 생성에 걸리는 시간을 비교하라. (네트워크 1회 추론 = 20ms 가정)
5. Flow matching으로 action chunk(5 step x 7 dim = 35차원 벡터)를 생성한다고 하자. 노이즈 z와 타겟 x_1은 각각 무엇인가?

---
