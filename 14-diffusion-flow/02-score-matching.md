# Score Matching

## VLA 연결고리

Score function은 diffusion model의 이론적 핵심이다. Diffusion Policy가 로봇 행동을 생성할 때, 실제로는 action 분포의 score를 학습하는 것이다. 또한 **Classifier-Free Guidance**(CFG)는 VLA에서 언어 지시나 이미지 조건에 따라 행동 생성을 제어하는 데 활용된다. pi-0는 CFG를 통해 언어 명령에 더 충실한 action을 생성한다.

---

## 핵심 개념

### 1. Score Function이란

데이터 분포 p(x)의 로그 확률의 기울기(gradient)이다.

$$s(x) = \nabla_x \log p(x)$$

**직관적 의미**: 현재 위치 x에서 데이터 밀도가 높아지는 방향을 가리키는 벡터이다.

- 데이터가 많은 곳(고밀도 영역)을 향해 화살표가 가리킨다
- 데이터가 적은 곳에서는 데이터가 많은 곳으로 향하는 큰 화살표
- 데이터 밀도의 정점(mode)에서는 화살표 크기가 0

**왜 유용한가**: $p(x)$ 자체를 알기는 매우 어렵다 (정규화 상수 계산이 불가능). 하지만 score는 정규화 상수와 무관하므로 학습이 가능하다.

### 2. Score와 Diffusion의 연결

DDPM에서 noise prediction과 score prediction은 수학적으로 동등하다.

**연결 관계**:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \cdot s_\theta(x_t, t)$$

즉, "노이즈를 예측하는 것"은 "노이즈가 섞인 데이터 분포의 score를 예측하는 것"과 같다.

**해석**: Reverse process에서 network는 "데이터 밀도가 높아지는 방향"으로 안내한다. 노이즈를 따라가면 데이터가 없는 곳으로 가고, score를 따라가면 데이터가 있는 곳으로 간다. 노이즈를 빼는 것과 score를 따라가는 것은 같은 행위이다.

### 3. Score Matching 학습

Score function을 neural network로 직접 학습하는 방법이다.

**Denoising Score Matching (DSM)**:
1. 깨끗한 데이터 $x_0$에 노이즈를 추가하여 $x_t$를 만든다
2. Network가 $x_t$에서의 score를 예측한다
3. 정답은 알려져 있다: 가우시안 노이즈를 추가했으므로 조건부 score를 계산할 수 있다
4. 예측과 정답의 차이를 최소화한다

이것이 DDPM의 epsilon-prediction과 본질적으로 같은 학습이다. 관점만 다르다.

**다양한 노이즈 레벨에서 학습**:
- 노이즈가 적을 때: 세밀한 구조 학습 (fine details)
- 노이즈가 많을 때: 전체적인 구조 학습 (global structure)
- 다양한 노이즈 레벨에서 동시에 학습함으로써 모든 스케일의 정보를 포착

### 4. Conditional Generation과 Guidance

조건(condition)에 맞는 데이터를 생성하는 방법이다.

**Conditional score**: 조건 $c$가 주어졌을 때의 score

$$s(x_t \mid c) = \nabla_{x_t} \log p(x_t \mid c)$$

**Classifier Guidance** (초기 방법):
- 별도의 classifier $p(c \mid x_t)$를 학습한다
- 생성 시 classifier의 gradient를 score에 더한다

$$s_{\text{guided}} = s(x_t) + w \cdot \nabla_{x_t} \log p(c \mid x_t)$$

- $w$가 클수록 조건에 더 충실하지만 다양성 감소
- 단점: 별도의 noisy classifier를 학습해야 한다

### 5. Classifier-Free Guidance (CFG)

별도의 classifier 없이 조건부 생성을 강화하는 방법이다. 현재 가장 널리 쓰인다.

**핵심 아이디어**: 하나의 network에서 조건부 score와 비조건부 score를 모두 학습한다.

**학습 시**: 일정 확률(예: 10%)로 조건 $c$를 빈 값(null)으로 대체한다.
- 조건이 있으면: $\epsilon_\theta(x_t, t, c)$ 학습
- 조건이 없으면: $\epsilon_\theta(x_t, t, \varnothing)$ 학습

**생성 시**: 두 예측을 조합한다.

$$\epsilon_{\text{guided}} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \left( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing) \right)$$

- $w = 1$: 일반 조건부 생성
- $w > 1$: 조건에 더 강하게 충실 (품질 향상, 다양성 감소)
- $w = 0$: 비조건부 생성 (조건 무시)

**VLA에서의 CFG**:
- 조건 c = 언어 지시문 ("pick up the red cup") + 카메라 이미지
- Guidance weight w를 높이면 언어 명령에 더 충실한 action 생성
- pi-0는 이 기법으로 task instruction에 맞는 행동을 강화한다

### 6. Score 관점의 통합적 이해

| 관점 | 학습 대상 | 생성 방법 |
|------|-----------|-----------|
| DDPM | 노이즈 $\epsilon$ | 노이즈를 점진적으로 제거 |
| Score matching | Score $\nabla \log p$ | Score 방향으로 이동 |
| SDE (확률미분방정식) | Drift + diffusion | SDE를 역방향으로 풀기 |

세 관점 모두 같은 것을 다르게 표현한다. Score 관점은 특히 이론적 분석과 새로운 샘플링 방법 개발에 유용하다. Flow matching(다음 다음 노트)도 이 연장선에 있다.

---

## 연습 주제 (코드 없이)

1. 2D 가우시안 혼합 분포(두 개의 봉우리)에서 score 벡터장을 상상하라. 두 봉우리 사이의 점에서 score는 어디를 가리키는가?
2. Noise prediction과 score prediction이 상수 배만큼 다른 이유를 직관적으로 설명하라.
3. Classifier-Free Guidance에서 w=0, w=1, w=5일 때 각각 어떤 생성 결과가 나올지 상상하라.
4. VLA에서 "pick up the red cup"이라는 조건으로 CFG를 적용한다고 하자. w가 너무 높으면 어떤 문제가 생길 수 있는가? (힌트: 다양성 부족과 과도한 "자신감")
5. 학습 시 조건을 랜덤하게 drop하는 것(CFG의 학습 트릭)이 왜 비조건부 모델도 동시에 학습하는 효과를 내는지 설명하라.

---

## 다음 노트

[Flow Matching](./03-flow-matching.md)
