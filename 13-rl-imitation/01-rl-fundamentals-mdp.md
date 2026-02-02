# RL Fundamentals: MDP

## VLA 연결고리

VLA(Vision-Language-Action) 모델은 로봇이 시각 정보와 언어 명령을 받아 행동을 출력하는 시스템이다. 강화학습(RL)은 이런 순차적 의사결정의 이론적 토대를 제공한다. 다만 실제 VLA 훈련에서는 RL보다 **Imitation Learning**을 주로 사용한다. 그 이유를 이해하려면 먼저 RL의 기본 구조를 알아야 한다.

---

## 핵심 개념

### 1. Markov Decision Process (MDP)

순차적 의사결정 문제를 수학적으로 정의하는 프레임워크다. 5개 요소로 구성된다.

| 요소 | 기호 | 의미 | 로봇 예시 |
|------|------|------|-----------|
| **State** | $s$ | 환경의 현재 상태 | 카메라 이미지, 관절 각도 |
| **Action** | $a$ | 에이전트가 취하는 행동 | 관절 토크, 그리퍼 열기/닫기 |
| **Reward** | $r$ | 행동에 대한 보상 신호 | 물건을 잡으면 +1 |
| **Transition** | $T(s' \mid s, a)$ | 다음 상태로의 전이 확률 | 물리 법칙에 따른 변화 |
| **Discount factor** | $\gamma$ | 미래 보상의 할인율 | 0.99 (먼 미래도 중요) |

**Markov Property**: 다음 상태는 오직 현재 상태와 행동에만 의존한다. 과거 이력은 필요 없다.

### 2. Policy (정책)

State를 입력받아 Action을 출력하는 함수다.

- **Deterministic policy**: $\pi(s) = a$ (하나의 행동을 확정적으로 출력)
- **Stochastic policy**: $\pi(a \mid s)$ (행동에 대한 확률분포를 출력)

VLA 모델은 본질적으로 policy다. 이미지와 언어를 state로 받고, 로봇 행동을 action으로 출력한다.

### 3. Value Function (가치 함수)

특정 상태(또는 상태-행동 쌍)에서 기대할 수 있는 미래 누적 보상이다.

- **State value** $V(s)$: 상태 $s$에서 시작했을 때 받을 총 보상의 기대값
- **Action value** $Q(s,a)$: 상태 $s$에서 행동 $a$를 취한 뒤 받을 총 보상의 기대값

최적의 policy는 모든 상태에서 가장 높은 value를 만드는 policy다.

### 4. Exploration vs Exploitation

- **Exploration**: 새로운 행동을 시도해서 더 좋은 전략을 발견
- **Exploitation**: 현재까지 알려진 최선의 행동을 반복

이 딜레마가 RL의 핵심 어려움이다. 로봇에게 이것이 특히 문제가 되는 이유는, 실제 로봇이 탐색하면서 자신이나 환경을 파손할 수 있기 때문이다.

### 5. Gym Interface

OpenAI Gym(현재 Gymnasium)은 RL 환경의 표준 인터페이스다.

```
env.reset()       ->  초기 state 반환
env.step(action)  ->  (next_state, reward, done, info) 반환
```

이 간단한 루프가 모든 RL 실험의 기본이다. 로봇 시뮬레이션 환경도 같은 구조를 따른다.

### 6. VLA가 RL 대신 Imitation Learning을 쓰는 이유

| 문제점 | RL에서 | IL에서 |
|--------|--------|--------|
| **Reward 설계** | 매우 어려움 (물건 잡기의 보상은?) | 불필요 (시범 데이터만 있으면 됨) |
| **Sample efficiency** | 수백만 에피소드 필요 | 수백~수천 시범이면 충분 |
| **안전성** | 탐색 과정에서 위험 | 시범 범위 내에서 안전 |
| **실제 로봇** | 시뮬에서만 현실적 | 실제 로봇 데이터로 직접 학습 가능 |

RL은 이론적 기반을 제공하지만, 실제 VLA는 사람이 보여준 시범(demonstration)에서 직접 행동을 배우는 방식을 택한다.

---

## 연습 주제 (코드 없이)

1. 로봇이 테이블 위의 컵을 잡는 과제를 MDP로 정의해 보라. State, Action, Reward, Transition을 각각 구체적으로 적어 보자.
2. "컵을 선반에 올려놓기" 과제의 reward를 설계한다고 할 때, sparse reward(성공/실패만)와 dense reward(거리 기반 등)의 장단점을 비교하라.
3. Stochastic policy가 deterministic policy보다 유리한 상황을 하나 생각해 보라.
4. Exploration 중 로봇 팔이 테이블을 강하게 치는 상황을 상상하라. 이것이 simulation-to-real(sim-to-real) transfer가 필요한 이유를 설명해 보라.
5. Gym interface의 step() 함수가 반환하는 4개 값 각각이 MDP의 어떤 요소에 대응되는지 매핑하라.

---

## 다음 노트

[Policy Gradient / PPO](./02-policy-gradient-ppo.md)
