# Diffusion for Actions

## VLA 연결고리

이 노트는 앞서 배운 diffusion/flow matching 이론을 **로봇 행동 생성에 직접 적용**하는 방법을 다룬다. Diffusion Policy, pi-0의 flow matching action head, 그리고 Transfusion 학습 방식이 핵심이다. 이것이 최신 VLA의 action 출력 메커니즘이다.

---

## 핵심 개념

### 1. 왜 Regression이 아닌 Diffusion/Flow인가

일반적인 BC는 observation이 주어졌을 때 action을 regression으로 예측한다 (MSE loss). 그런데 이 방식에는 치명적인 문제가 있다.

**Multimodal Action Distribution 문제**:
```
같은 관찰(컵이 앞에 있음)에 대해:
- 전문가 A: 왼쪽으로 돌아서 잡음
- 전문가 B: 오른쪽으로 돌아서 잡음

MSE regression 결과: 가운데로 직진 (둘의 평균) -> 실패!
```

MSE는 모든 정답의 평균을 출력한다. 하지만 평균 행동은 어떤 전문가의 행동과도 다르며, 종종 물리적으로 불가능하다.

**Diffusion/Flow의 해결책**: 행동의 전체 분포를 학습하고, 그 분포에서 샘플링한다. "왼쪽" 또는 "오른쪽" 중 하나를 선택하지, 평균인 "가운데"를 출력하지 않는다.

### 2. Diffusion Policy

2023년 Chi et al.이 제안한, diffusion으로 로봇 행동을 생성하는 방법이다.

**구조**:
```
입력: 관찰(observation) = 카메라 이미지 + proprioception
출력: action chunk = 여러 time step의 연속 행동

생성 과정:
1. 랜덤 노이즈에서 시작 (action 차원의 가우시안 노이즈)
2. 관찰을 조건으로, denoising을 반복
3. 최종 결과: 깨끗한 action chunk
```

**Action Chunk**: 한 번에 여러 미래 step의 행동을 생성한다.
- 예: 16 step x 7 dim = 112차원 벡터를 한 번에 생성
- 장점: 시간적으로 일관된 부드러운 동작
- 장점: VLA의 느린 추론을 보상 (한 번 추론으로 여러 step 실행)

**Diffusion Policy의 성과**:
- MSE regression보다 일관되게 높은 성공률
- 다양한 manipulation 과제에서 SOTA
- 멀티모달 행동을 정확하게 생성

### 3. Action Chunk의 실행 전략

한 번에 여러 step을 예측하므로, 실행 전략이 필요하다.

**Receding Horizon 방식**:
1. Time step t에서 action chunk [a_t, a_{t+1}, ..., a_{t+H-1}]를 생성한다 (H = horizon)
2. 처음 K개만 실행한다 (K < H)
3. K step 후 새로운 관찰로 다시 action chunk를 생성한다
4. 반복한다

```
생성: [a0, a1, a2, a3, a4, a5, a6, a7]  (H=8)
실행: [a0, a1, a2, a3]                    (K=4만 실행)
생성: [a4', a5', a6', a7', a8', a9', a10', a11']  (새 관찰 기반)
실행: [a4', a5', a6', a7']
...
```

이 방식으로 관찰을 주기적으로 반영하면서도 부드러운 동작을 유지한다.

### 4. pi-0: VLM + Flow Matching

Physical Intelligence의 pi-0(pi-zero)는 **VLM과 flow matching을 결합한 VLA**이다.

**아키텍처**:
```
입력:
  - 카메라 이미지 (시각)
  - 언어 지시문 ("pick up the red cup")
  - Proprioception (로봇 상태)
  - 노이즈가 섞인 action chunk (flow matching 입력)

처리:
  VLM backbone (PaliGemma 기반)이 모든 입력을 joint attention으로 처리

출력:
  Denoised action chunk (flow matching으로 노이즈 제거)
```

**핵심 설계 결정**:
- Language/vision 토큰과 action 토큰이 **같은 transformer** 안에서 attention을 공유한다
- Action은 연속값으로, flow matching의 velocity prediction으로 생성된다
- 언어 이해와 행동 생성이 하나의 모델에서 end-to-end로 이루어진다

**학습 과정**:
1. 사전학습된 VLM에서 시작
2. 로봇 데이터에서 flow matching loss로 fine-tuning
3. 언어 토큰에는 cross-entropy loss, action 토큰에는 flow matching loss를 동시 적용

### 5. Transfusion 학습

하나의 모델에서 이산 토큰(언어)과 연속 값(행동)을 동시에 학습하는 방식이다.

**문제**: LLM은 cross-entropy loss로 학습하고, diffusion/flow는 MSE류 loss로 학습한다. 하나의 모델에서 두 종류의 출력을 어떻게 학습하는가?

**Transfusion의 해결책**: 입력 modality에 따라 다른 loss를 적용한다.

```
총 Loss = L_language + lambda * L_action

L_language = cross-entropy (이산 토큰 예측, 언어 부분)
L_action   = flow matching loss (연속 값 예측, action 부분)
```

| 모달리티 | 표현 | Loss | 생성 방법 |
|----------|------|------|-----------|
| 텍스트 | 이산 토큰 | Cross-entropy | Autoregressive decoding |
| 이미지 | 연속 패치 | Diffusion/Flow loss | Denoising |
| **로봇 행동** | 연속 벡터 | Flow matching loss | Flow matching |

**의의**: VLA는 본질적으로 multimodal 모델이며, Transfusion은 각 modality에 가장 적합한 학습 방식을 자연스럽게 결합한다. pi-0가 이 접근을 채택한다.

### 6. Diffusion/Flow 기반 Action 생성 정리

| 시스템 | 생성 방법 | Action 형태 | 특징 |
|--------|-----------|------------|------|
| Diffusion Policy | DDPM | Chunk (연속) | 멀티모달 분포 학습의 원조 |
| Octo | Diffusion | Chunk (연속) | 범용 로봇 policy |
| **pi-0** | Flow matching | Chunk (연속) | VLM + flow, Transfusion |
| **pi-0.5** | Flow matching | Chunk (연속) | 더 큰 스케일, 더 많은 로봇 |
| **SmolVLA** | Flow matching | Chunk (연속) | 경량화, 오픈소스 |
| **GR00T N1** | Flow matching | Chunk (연속) | 휴머노이드 특화 |

2025~2026년 기준으로, **Flow matching + action chunk**가 VLA action head의 주류가 되고 있다.

---

## 연습 주제 (코드 없이)

1. "컵을 왼쪽 또는 오른쪽으로 돌아서 잡기" 과제에서, MSE regression, discrete tokenization, diffusion 세 방법이 각각 어떤 action을 출력할지 비교하라.
2. Action chunk horizon H=16, 실행 step K=4일 때, 1초 동안(10Hz 제어) 몇 번의 diffusion/flow 추론이 필요한지 계산하라.
3. pi-0에서 language 토큰과 action 토큰이 같은 transformer에서 attention을 공유하는 것의 이점을 설명하라. (힌트: "빨간 컵"이라는 언어 정보가 action 생성에 직접 영향)
4. Transfusion에서 lambda(action loss의 가중치)가 너무 크거나 작으면 각각 어떤 문제가 생기는지 생각하라.
5. Flow matching이 diffusion보다 적은 step으로 action을 생성할 수 있다. 이것이 실시간 로봇 제어에서 왜 결정적으로 중요한지 구체적 숫자로 설명하라 (예: 추론 시간 vs 제어 주파수).

---

## 다음 노트

[VLA 폴더로 이동](../15-vla/)
