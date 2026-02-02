# Imitation Learning / Behavior Cloning

## VLA 연결고리

Behavior Cloning(BC)은 **VLA의 핵심 훈련 방법**이다. VLA는 사람이 로봇을 조종하며 수집한 시범(demonstration) 데이터에서, 관찰(observation)을 입력으로 받고 행동(action)을 출력하도록 지도학습한다. OpenVLA, RT-2, pi-0 모두 BC를 기반으로 훈련된다.

---

## 핵심 개념

### 1. Imitation Learning이란

전문가(expert)의 행동을 모방하여 policy를 학습하는 방법이다. Reward 함수를 설계할 필요가 없다.

**RL과의 결정적 차이**:

| 항목 | RL | Imitation Learning |
|------|----|--------------------|
| 학습 신호 | Reward (환경이 제공) | 전문가 시범 (사람이 제공) |
| 탐색 필요 | 필수 (exploration) | 불필요 |
| 데이터 | 에이전트가 직접 수집 | 전문가가 미리 수집 |
| 안전성 | 탐색 중 위험 가능 | 시범 데이터 범위 내 |

### 2. Behavior Cloning (BC)

가장 단순하고 강력한 Imitation Learning 방법이다.

**핵심 아이디어**: 전문가의 (observation, action) 쌍을 수집하고, observation -> action 매핑을 지도학습한다. 분류(classification) 또는 회귀(regression) 문제로 환원된다.

**BC의 과정**:
1. 전문가가 과제를 수행하며 데이터를 수집한다
   - 관찰(obs): 카메라 이미지, 로봇 관절 상태 등
   - 행동(act): 전문가가 취한 행동 (관절 속도, 그리퍼 명령 등)
2. Dataset D = {(o_1, a_1), (o_2, a_2), ..., (o_N, a_N)}을 구성한다
3. Policy network pi_theta(a|o)를 D에 대해 지도학습한다
   - 연속 행동: MSE loss (예측 행동 vs 실제 행동)
   - 이산 행동: Cross-entropy loss

**VLA에서의 BC**:
- Observation = 카메라 이미지 + 언어 지시문 + 로봇 proprioception
- Action = 로봇 행동 (7-DoF end-effector 명령 등)
- Model = Vision-Language Model을 backbone으로 사용
- Loss = 행동 토큰에 대한 cross-entropy 또는 MSE

### 3. Distribution Shift 문제

BC의 가장 큰 약점이다.

**문제**: 학습 시에는 전문가의 관찰만 본다. 그런데 실행 시 작은 실수가 누적되면 전문가가 절대 가지 않았을 상태에 도달한다. 그 상태에서 어떻게 행동해야 하는지 배운 적이 없으므로 더 큰 실수를 하고, 이것이 연쇄적으로 커진다.

```
전문가 경로:  A -> B -> C -> D -> 성공
BC 실행:      A -> B -> B' -> ??? -> 실패
                         ^
                   작은 오차 발생, 이후 학습 범위 벗어남
```

이것을 **compounding error** 또는 **covariate shift**라고 부른다.

**완화 방법들**:
- 더 많은 데이터 수집 (다양한 상황 커버)
- 데이터 증강 (augmentation)
- Action chunking (한 번에 여러 스텝의 행동을 예측)
- 학습 시 노이즈 주입

### 4. DAgger (Dataset Aggregation)

Distribution shift를 체계적으로 해결하는 방법이다.

**과정**:
1. 전문가 데이터로 초기 policy를 BC로 학습한다
2. 학습된 policy로 실행하며 방문하는 상태를 기록한다
3. 그 상태들에 대해 전문가가 올바른 행동을 레이블링한다
4. 새 데이터를 기존 데이터셋에 합치고 다시 학습한다
5. 반복한다

**장점**: 이론적으로 distribution shift를 해결할 수 있다.

**한계**: 매번 전문가의 개입이 필요하다. 실제 로봇에서는 비용이 크다.

### 5. BC가 VLA에서 작동하는 이유

이론적으로 BC는 distribution shift 문제가 있지만, 최근 VLA에서 잘 작동하는 이유가 있다.

| 요인 | 설명 |
|------|------|
| **대규모 데이터** | 수만~수십만 에피소드의 다양한 시범 |
| **강력한 backbone** | 사전학습된 VLM이 일반화 능력 제공 |
| **Action chunking** | 여러 스텝을 한 번에 예측하여 오차 누적 감소 |
| **다양한 환경** | 다양한 로봇, 과제, 환경에서 수집 |
| **데이터 증강** | 이미지 augmentation으로 robustness 향상 |

결국 "충분히 많고 다양한 데이터 + 충분히 강력한 모델"이면 BC만으로도 뛰어난 성능을 낸다.

### 6. Imitation Learning의 다른 방법들

| 방법 | 핵심 아이디어 | 비고 |
|------|--------------|------|
| **BC** | 직접 지도학습 | VLA의 주력 |
| DAgger | 반복적 데이터 수집 | 전문가 개입 필요 |
| IRL (Inverse RL) | 시범에서 reward 함수를 추론 | 계산 비용 큼 |
| GAIL | GAN으로 전문가 행동 모방 | 학습 불안정 |

---

## 연습 주제 (코드 없이)

1. "컵 잡기" 과제에 BC를 적용한다고 할 때, observation과 action을 구체적으로 정의해 보라.
2. Distribution shift가 발생하는 구체적인 시나리오를 하나 만들어 보라. 로봇이 어떤 실수를 하고 어떻게 실패로 이어지는지 서술하라.
3. BC를 분류 문제로 푸는 것과 회귀 문제로 푸는 것의 차이를 설명하라. VLA에서 action tokenization이 어떤 선택에 해당하는지 연결하라.
4. DAgger의 3단계를 "접시 정리" 과제에 적용해 보라. 전문가는 어떤 상황에서 어떤 레이블을 제공해야 하는가?
5. BC 데이터셋에서 같은 상태에 대해 전문가마다 다른 행동을 보이면(multimodal action distribution) 어떤 문제가 생기는지 설명하라. (힌트: MSE loss의 한계와 Diffusion Policy의 동기)

---
