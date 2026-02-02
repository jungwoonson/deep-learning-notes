# Action Tokenization

## VLA 연결고리

VLA 모델은 LLM을 backbone으로 사용한다. LLM은 이산적인 토큰(discrete token)을 출력하도록 설계되었다. 그런데 로봇 행동은 연속적인 실수값(continuous value)이다. 이 간극을 메우는 것이 **action tokenization**이다. OpenVLA는 256-bin discretization을, 최신 FAST tokenizer는 DCT 압축을 사용한다. 이 설계 선택이 VLA의 성능과 효율성을 크게 좌우한다.

---

## 핵심 개념

### 1. 왜 연속 행동을 이산화하는가

**LLM의 구조적 제약**:
- LLM의 출력 레이어는 vocabulary에 대한 softmax이다
- 고정된 크기의 어휘에서 토큰을 하나씩 선택하는 구조이다
- 연속값 0.0372를 직접 출력할 수 없다

**이산화의 장점**:
- LLM의 기존 아키텍처를 그대로 활용 가능
- Cross-entropy loss 사용 가능 (분류 문제로 환원)
- 멀티모달 출력 분포를 자연스럽게 표현 가능 (MSE는 평균으로 붕괴)
- 언어 토큰과 행동 토큰을 동일한 방식으로 처리 가능

### 2. 256-bin Discretization (OpenVLA)

OpenVLA가 사용하는 방법이다.

**과정**:
1. 각 action 차원을 학습 데이터의 통계로 정규화한다
2. 정규화된 범위를 256개 균등 구간(bin)으로 나눈다
3. 연속값을 가장 가까운 bin 인덱스로 변환한다
4. 이 인덱스가 LLM vocabulary의 새로운 토큰이 된다

**예시** (하나의 action 차원):

```
연속값 범위: [-1.0, +1.0]
256개 bin:   bin_0=[-1.0, -0.992], bin_1=[-0.992, -0.984], ..., bin_255=[0.992, 1.0]
연속값 0.5 -> bin_191에 매핑 -> 토큰 "action_191" 출력
```

**7-DoF action의 경우**:
- 각 차원마다 256개 bin -> 7개 토큰을 순차적으로 출력
- 총 vocabulary 확장: 기존 LLM vocab + 256개 action 토큰
- 하나의 행동을 생성하려면 7번의 autoregressive decoding이 필요

**해상도 문제**: 256개 bin은 각 차원에서 약 0.008 단위의 해상도를 제공한다. 정밀한 조작에는 부족할 수 있다. bin 수를 늘리면 해상도가 올라가지만 학습 난이도도 올라간다.

### 3. LLM Vocabulary 확장

기존 LLM의 토큰 어휘(vocabulary)에 action 토큰을 추가하는 방법이다.

**구조**:
```
기존 LLM vocabulary:  [token_0, token_1, ..., token_32000]  (언어 토큰)
확장 후:              [token_0, ..., token_32000, act_0, act_1, ..., act_255]
```

- Embedding layer와 output layer에 새로운 행이 추가된다
- 새 토큰의 embedding은 random 초기화 후 fine-tuning된다
- 언어 토큰과 action 토큰이 동일한 transformer를 통해 처리된다

**이것의 의미**: VLA는 "언어를 말하듯이 행동을 말한다". 텍스트 생성과 동일한 메커니즘으로 로봇 행동을 생성한다.

### 4. FAST Tokenizer (DCT 압축)

2024년에 제안된 더 효율적인 action tokenization 방법이다.

**기존 방식의 문제**:
- 7-DoF action을 7개 토큰으로 표현 -> 느린 추론
- Action chunk (여러 스텝)을 표현하려면 토큰 수가 급증
  - 예: 10 스텝 x 7 차원 = 70개 토큰을 순차 생성

**FAST의 핵심 아이디어**:

1. **Action chunk 수집**: 여러 연속 time step의 action을 묶는다
   - 예: 5 스텝 x 7 차원 = 35개 연속값

2. **DCT (Discrete Cosine Transform)**: 시계열 action 데이터를 주파수 영역으로 변환한다
   - 로봇 행동은 대부분 저주파(부드러운 움직임)이므로 고주파 성분을 버릴 수 있다
   - 데이터 압축 효과: 35개 값 -> 소수의 주요 DCT 계수만 유지

3. **Tokenization**: 압축된 DCT 계수를 이산 토큰으로 변환한다
   - 결과: 35개 값이 훨씬 적은 수의 토큰(예: 8~16개)으로 표현된다

**장점**:
- 토큰 수 대폭 감소 -> 추론 속도 향상
- 시계열의 부드러움(smoothness)을 자연스럽게 반영
- Action chunk를 효율적으로 인코딩

### 5. Discrete vs Continuous 비교

action을 이산화하지 않고 연속값으로 직접 출력하는 방법도 있다.

| 측면 | Discrete (토큰화) | Continuous (연속 출력) |
|------|-------------------|----------------------|
| **아키텍처** | LLM 그대로 사용 | Regression head 추가 필요 |
| **Loss** | Cross-entropy | MSE 또는 diffusion loss |
| **멀티모달 분포** | 자연스럽게 표현 | MSE는 불가, diffusion은 가능 |
| **해상도** | bin 수에 제한 | 이론적으로 무한 |
| **대표 모델** | OpenVLA, RT-2 | Diffusion Policy, pi-0, Octo |
| **최신 추세** | FAST tokenizer로 개선 | Flow matching으로 발전 |

**최근 흐름**: 두 접근이 공존하지만, Diffusion/Flow matching 기반의 연속 출력 방식이 점점 더 주목받고 있다 (pi-0, SmolVLA 등). 한편 FAST tokenizer는 이산 방식의 효율을 크게 개선했다.

### 6. Action Tokenization 설계 요약

| 모델 | 방법 | Action 차원 | 토큰 수/스텝 |
|------|------|------------|-------------|
| RT-2 | 256-bin per dim | 7 | 7 |
| OpenVLA | 256-bin per dim | 7 | 7 |
| FAST (SmolVLA 등) | DCT + tokenize | 7 x chunk | 8~16 (chunk 전체) |
| pi-0 | Flow matching (연속) | 7 x chunk | N/A (연속값) |
| Octo | Diffusion (연속) | 7 x chunk | N/A (연속값) |

---

## 연습 주제 (코드 없이)

1. 256-bin discretization에서 action 범위가 [-1, 1]일 때, 하나의 bin이 커버하는 범위를 계산하라. 이 해상도가 정밀 조작(예: 나사 돌리기)에 충분한지 논의하라.
2. LLM이 7개 action 토큰을 순차적으로 생성할 때, 첫 번째 차원의 예측이 틀리면 나머지 차원에 어떤 영향이 있을지 생각하라.
3. FAST tokenizer의 DCT 압축에서 고주파 성분을 버리는 것이 왜 로봇 행동에서 합리적인지 설명하라. (힌트: 로봇 팔은 갑자기 방향을 바꾸지 않는다)
4. Multimodal action distribution 문제를 이산화(discrete)와 diffusion이 각각 어떻게 해결하는지 비교하라.
5. Action chunk 크기가 커질수록(예: 5 -> 20 스텝) 256-bin 방식과 FAST 방식의 토큰 수 차이가 어떻게 변하는지 계산하라.

---

## 다음 노트

[Diffusion Fundamentals](../14-diffusion-flow/01-diffusion-fundamentals.md)
