# Policy Gradient / PPO

## VLA 연결고리

VLA 모델 자체는 Imitation Learning으로 훈련되지만, VLA의 기반이 되는 LLM은 **PPO 기반 RLHF**로 정렬(alignment)된다. PPO는 현재 AI에서 가장 널리 쓰이는 RL 알고리즘이며, policy를 직접 최적화한다는 아이디어가 핵심이다.

---

## 핵심 개념

### 1. Policy Gradient의 핵심 아이디어

Value function을 먼저 배우고 그로부터 policy를 유도하는 대신, **policy 자체를 직접 최적화**한다.

목표: 기대 누적 보상 J(theta) = E[sum of rewards]를 최대화하는 policy parameter theta를 찾는 것.

핵심 통찰: 좋은 결과를 낸 행동의 확률은 올리고, 나쁜 결과를 낸 행동의 확률은 낮춘다.

### 2. REINFORCE

가장 기본적인 policy gradient 알고리즘이다.

**동작 원리**:
1. 현재 policy로 에피소드를 끝까지 진행한다
2. 각 행동에 대해, 그 이후 받은 총 보상(return)을 계산한다
3. Return이 높은 행동의 확률을 올리는 방향으로 policy를 업데이트한다

**한계**:
- **High variance**: 한 에피소드의 return이 우연히 높거나 낮을 수 있다
- **Sample inefficiency**: 한 에피소드를 다 끝내야 한 번 업데이트 가능
- Baseline을 빼서 variance를 줄일 수 있다 (return - baseline)

### 3. Advantage Function

단순히 "좋았다/나빴다"보다 "평균 대비 얼마나 좋았는가"를 측정한다.

```
A(s, a) = Q(s, a) - V(s)
```

- Q(s,a): 이 상태에서 이 행동을 했을 때의 가치
- V(s): 이 상태의 평균적 가치
- A > 0: 평균보다 좋은 행동 (확률 높이기)
- A < 0: 평균보다 나쁜 행동 (확률 낮추기)

Advantage를 사용하면 variance가 크게 줄어들어 학습이 안정된다.

### 4. PPO (Proximal Policy Optimization)

현재 가장 널리 쓰이는 policy gradient 알고리즘이다. 2017년 OpenAI에서 제안했다.

**PPO가 해결하는 문제**: 기존 policy gradient는 한 번의 업데이트가 너무 크면 policy가 갑자기 나빠질 수 있다. PPO는 업데이트 크기를 제한하여 안정적 학습을 보장한다.

**Clipping 메커니즘**:

PPO는 새 policy와 이전 policy의 확률 비율(ratio)을 계산한다:

```
r(theta) = pi_new(a|s) / pi_old(a|s)
```

이 ratio가 1에서 너무 벗어나지 않도록 clip한다:

```
L = min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)
```

- epsilon은 보통 0.2
- Advantage가 양수이면: ratio가 1.2를 넘어도 1.2로 잘림 (너무 많이 올리지 않음)
- Advantage가 음수이면: ratio가 0.8 아래로 가도 0.8로 잘림 (너무 많이 내리지 않음)

**결과**: 한 번에 조금씩만 바꾸므로 학습이 안정적이다.

### 5. PPO가 RLHF에서 쓰이는 이유

LLM의 RLHF(Reinforcement Learning from Human Feedback) 과정:

| 단계 | 내용 |
|------|------|
| 1. SFT | 지도학습으로 기본 대화 능력 학습 |
| 2. Reward Model | 인간 선호도 데이터로 보상 모델 훈련 |
| 3. PPO 최적화 | Reward model의 점수를 높이도록 LLM policy 업데이트 |

PPO가 선택된 이유:
- **안정성**: Clipping으로 LLM이 갑자기 이상한 출력을 하는 것을 방지
- **KL penalty**: 원래 SFT 모델에서 너무 멀어지지 않도록 제한
- **실용성**: 구현이 비교적 단순하고 하이퍼파라미터에 덜 민감

이 기법이 GPT, Claude 같은 LLM을 "정렬"하는 데 핵심 역할을 했다. VLA의 language 부분도 RLHF된 LLM에서 시작한다.

### 6. Policy Gradient 계열 정리

| 알고리즘 | 핵심 특징 | 용도 |
|----------|-----------|------|
| REINFORCE | 가장 단순, high variance | 교육용, 간단한 문제 |
| A2C/A3C | Advantage + 병렬화 | 게임 AI |
| TRPO | Trust region으로 안정화 | 로봇 제어(연구) |
| **PPO** | Clipping으로 간단하게 안정화 | **RLHF, 범용** |
| SAC | 최대 엔트로피, off-policy | 로봇 연속 제어 |

---

## 연습 주제 (코드 없이)

1. REINFORCE에서 baseline을 빼는 것이 왜 variance를 줄이는지 직관적으로 설명해 보라. (힌트: 모든 return이 양수인 경우를 생각)
2. PPO clipping을 그림으로 그려보라. x축은 ratio r, y축은 clipped objective. Advantage > 0과 < 0 두 경우 모두 그려보자.
3. RLHF에서 KL penalty가 없으면 어떤 문제가 생길지 상상해 보라. (reward hacking)
4. LLM에서 "한 에피소드"란 무엇인가? State, action, reward를 LLM 대화 맥락에서 정의해 보라.
5. PPO가 on-policy인데도 sample efficient한 이유를 설명해 보라. (힌트: mini-batch로 여러 번 업데이트)

---

## 다음 노트

[Imitation Learning / Behavior Cloning](./03-imitation-learning-bc.md)
