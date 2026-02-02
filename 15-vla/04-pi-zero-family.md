# pi-zero 계열 (pi-0 Family)

## VLA와의 연결

**pi-0 계열은 VLA의 행동 생성 방식을 근본적으로 바꿨다.** RT-2와 OpenVLA가 행동을 이산 토큰(256 bins)으로 생성한 반면, pi-0는 **Flow Matching**을 사용하여 연속적이고 부드러운 행동을 직접 출력한다. 이전 커리큘럼의 Diffusion/Flow 파트(Part 14)에서 배운 Flow Matching이 여기서 실제 로봇 제어에 적용된다. pi-0에서 시작하여 pi-0-FAST, pi-0.5, pi-star-0.6까지 발전하며, "학습 → 범용화 → 자기 개선"의 경로를 보여준다.

---

## 핵심 개념

### 1. pi-0: Flow Matching 기반 VLA (2024.10)

#### 왜 Flow Matching인가

```
이산 토큰 방식 (RT-2, OpenVLA)의 한계:

1. 양자화 오류 (Quantization Error)
   연속값 0.3782 → bin 96 (= 0.376) → 오차 0.002
   7개 차원에 걸쳐 누적되면 의미있는 오차

2. 차원 간 독립 가정
   각 행동 차원을 독립적으로 토큰화
   실제로는 x, y, z가 서로 상관 관계를 가짐
   예: 대각선 움직임 = x와 y의 동시 변화

3. 불연속적 궤적
   이산 토큰 → 계단 형태의 행동
   부드러운 움직임이 어려움

Flow Matching의 장점:
   → 연속 분포에서 직접 샘플링
   → 다차원 상관관계를 자연스럽게 포착
   → 부드러운 연속 궤적 생성 가능
```

#### pi-0 아키텍처

```
pi-0 전체 구조:

카메라 이미지 + 언어 명령
         ↓
[사전학습 VLM (PaLiGemma 기반, 3B)]
         ↓
시각-언어 표현 (고수준 이해)
         ↓
[Flow Matching Action Expert]
         ↓
연속적 행동 벡터 출력

두 단계의 역할:
  VLM 부분:     "무엇을 해야 하는가?" (장면 이해 + 명령 해석)
  Flow 부분:    "어떻게 움직여야 하는가?" (구체적 행동 생성)
```

#### Flow Matching 복습 (Part 14 연결)

```
Flow Matching의 핵심 아이디어:

노이즈 분포 → 데이터 분포로의 "흐름"을 학습

pi-0에서의 적용:
  노이즈: 랜덤 행동 벡터 (가우시안 노이즈)
  데이터: 전문가의 실제 행동 (텔레오퍼레이션 데모)

  학습:
    1. 전문가 행동(target action)과 랜덤 노이즈를 선형 보간
       x_t = (1-t) * noise + t * target_action  (t는 0~1)
    2. 모델이 x_t에서 target_action 방향의 "벡터장"을 예측
    3. 이 벡터장을 따라가면 노이즈 → 올바른 행동

  추론:
    1. 랜덤 노이즈에서 시작
    2. 학습된 벡터장을 따라 소수 스텝(예: 10 스텝) 이동
    3. 최종 위치 = 예측된 행동

이산 토큰과의 비교:
  이산: "256개 선택지 중 하나를 고르시오" (분류 문제)
  Flow:  "노이즈에서 올바른 행동으로의 경로를 따라가시오" (회귀 문제)
```

#### pi-0의 핵심 성과

```
제어 주파수: 50Hz
  → OpenVLA (~5-10Hz) 대비 5~10배 빠름
  → 민첩한 조작(dexterous manipulation) 가능

7개 로봇 플랫폼:
  1인 로봇 팔, 양팔 로봇, 이동 로봇 등
  하나의 모델로 모두 제어

다양한 task:
  - 빨래 접기 (복잡한 천 조작)
  - 테이블 정리
  - 식기 세척기에 접시 넣기
  → 이전 VLA가 시도하지 못한 복잡한 task
```

### 2. pi-0-FAST (2025.06)

#### FAST 토크나이저의 도입

pi-0-FAST는 Flow Matching과 **Autoregressive 토큰 생성**을 결합한다:

```
pi-0의 접근:       Flow Matching (연속)
OpenVLA의 접근:    Autoregressive 토큰 (이산)
pi-0-FAST의 접근:  Autoregressive + FAST 토크나이저 (효율적 이산)

FAST (Fast Action Tokenization) 토크나이저:
  핵심: DCT(이산 코사인 변환)로 행동 시퀀스를 압축

  기존 방식:
    10 타임스텝 * 7 차원 = 70 토큰 (OpenVLA 방식)

  FAST 방식:
    10 타임스텝 * 7 차원의 행동 시퀀스
    → DCT 변환 (주파수 도메인으로)
    → 고주파 성분 제거 (압축)
    → 소수의 토큰으로 표현 (예: ~10-20 토큰)
    → 약 4-7배 토큰 수 감소!
```

#### 왜 FAST가 중요한가

```
학습 속도: 5배 빠름
  토큰 수 감소 → 시퀀스 길이 단축 → 학습/추론 모두 빠름

장점:
  1. Autoregressive 방식의 장점 유지
     - 확장성(scalability)이 입증된 방법
     - LLM 인프라를 그대로 활용 가능
  2. Flow Matching 수준의 행동 품질
     - DCT가 연속적 궤적의 부드러움을 보존
  3. 범용성
     - 다양한 로봇, 다양한 행동 차원에 적용 가능
```

### 3. pi-0.5 (2025.06)

```
pi-0.5의 핵심: Open-World Generalization

pi-0: 학습 환경에서 우수한 성능
pi-0.5: 학습에 없는 환경에서도 일반화!

새로운 능력:
  1. 새로운 주방에서도 동작
  2. 이전에 보지 못한 물체도 조작
  3. 자연어로 새로운 task를 지정하면 수행

핵심 요인:
  - 더 다양한 학습 데이터
  - 개선된 VLM 백본 (더 강한 세상 지식)
  - 데이터 증강(augmentation) 기법
```

### 4. pi-star-0.6: 자기 개선하는 VLA (2025.08)

#### RECAP: Reinforcement Learning from Robot Experience

```
기존 VLA 학습:
  사람이 텔레오퍼레이션으로 데모 수집 → 모방 학습(Imitation Learning)
  한계: 사람의 데모 데이터만큼만 학습 가능

pi-star-0.6의 혁신: RECAP
  Robot Experience from Continuous Autonomous Practice

  1단계: 초기 정책 학습 (기존 방식)
    사람의 데모 데이터로 pi-0 학습

  2단계: 자율 연습 (새로운 방식!)
    로봇이 스스로 task를 시도
    → 성공/실패를 자동 판별
    → 성공 경험을 학습 데이터에 추가
    → 실패에서도 부분적으로 학습

  3단계: RL 기반 개선
    보상 모델이 행동의 품질을 평가
    → 더 좋은 행동을 강화
    → 더 나쁜 행동을 억제
    → 점진적으로 성능 향상
```

#### 18시간 자율 운영의 의미

```
pi-star-0.6 실험:
  로봇이 18시간 동안 사람의 개입 없이 자율적으로:
  - task를 반복 시도
  - 성공/실패를 기록
  - 자신의 정책을 개선

결과:
  - 시간이 지남에 따라 성공률 상승
  - 사람의 추가 데모 없이 성능 개선
  - "자기 개선하는 로봇"의 최초 실질적 시연

VLA 발전 경로 전체 요약:
  모방 학습 (IL)    → "사람처럼 해라" (pi-0)
  + 자율 연습       → "스스로 연습해라" (pi-star-0.6, RECAP)
  + 강화 학습 (RL)  → "더 잘 해라" (보상 기반 개선)

이것은 Part 13(RL/Imitation)에서 배운 내용의 실현:
  Imitation Learning → Online RL → Self-Improvement
```

### 5. Physical Intelligence: 회사와 비전

```
Physical Intelligence (PI):
  설립: 2024년
  자금: $600M Series A (로보틱스 역대 최대)
  비전: "범용 로봇 파운데이션 모델"

창업자:
  - Karol Hausman (Google DeepMind RT 팀 리드)
  - Sergey Levine (UC Berkeley, 로봇 RL 선구자)
  - Chelsea Finn (Stanford, Meta-Learning 선구자)
  - Brian Ichter (Google DeepMind)

왜 주목하는가:
  1. RT-1/RT-2를 만든 핵심 인물들의 합류
  2. 학계 최고 수준의 로봇 학습 전문가
  3. $600M으로 대규모 데이터 수집 + 컴퓨팅 가능
  4. pi-0 → pi-0-FAST → pi-0.5 → pi-star-0.6의 빠른 진화

PI의 접근 방식:
  "충분히 크고 다양한 데이터 + 강력한 모델 = 범용 로봇"
  이것은 LLM의 "Scaling Law"를 로봇에 적용하려는 시도
```

### 6. pi-0 계열 비교표

```
          pi-0           pi-0-FAST      pi-0.5         pi-star-0.6
──────────────────────────────────────────────────────────────────────
시기      2024.10        2025.06        2025.06        2025.08
행동 방식  Flow Matching  Autoregressive  Flow Matching  Flow + RL
                        + FAST token
제어 Hz   50Hz          ~30-50Hz       50Hz           50Hz
학습 방식  IL(모방)       IL             IL             IL + RL(RECAP)
일반화     학습 환경      학습 환경       새로운 환경!    새로운 환경
자기개선   없음           없음            없음            있음! (18hr)
학습 속도  기준           5배 빠름        기준           기준+RL
핵심 기여  연속행동VLA    효율적토큰화    범용일반화     자기개선VLA
```

---

## 연습 주제 (코드 없이 생각해보기)

1. **Flow Matching vs 이산 토큰**: 로봇이 직선으로 10cm 이동해야 할 때, 256 bins 이산 방식과 Flow Matching 연속 방식의 결과 궤적을 각각 그려보라. 어떤 방식이 더 부드러운 움직임을 생성하는가?

2. **DCT 압축의 직관**: DCT(이산 코사인 변환)가 행동 시퀀스를 압축할 수 있는 이유를 생각해보라. (힌트: JPEG 이미지 압축도 DCT를 사용한다. 로봇 행동 시퀀스에서 "고주파 성분"은 무엇을 의미하는가?)

3. **50Hz의 의미**: 50Hz 제어란 0.02초마다 새로운 행동을 생성한다는 뜻이다. 이것이 "빨래 접기" 같은 복잡한 task에서 왜 중요한지, 5Hz(0.2초마다)와 비교하여 설명하라.

4. **RECAP의 보상 설계**: pi-star-0.6에서 로봇이 자율적으로 "성공/실패"를 판단한다. "컵을 집어서 선반에 놓기" task에서 성공/실패를 자동 판별하는 방법을 설계해보라. (힌트: 카메라 이미지에서 물체 위치 확인)

5. **Scaling Law의 로봇 적용**: LLM에서는 "모델 크기 x 데이터 x 컴퓨팅 → 성능"이라는 Scaling Law가 성립한다. 로봇 VLA에서도 같은 법칙이 적용될까? 로봇 데이터 수집의 비용이 텍스트 데이터 수집과 어떻게 다른지 고려하라.

6. **모방 학습의 천장**: 모방 학습은 "전문가만큼만" 할 수 있다는 근본적 한계가 있다. pi-star-0.6의 RL이 이 천장을 어떻게 돌파하는지 설명하라. (Part 13 RL/Imitation 복습)

---

## 다음 노트

**다음**: [Dual-System 아키텍처](./05-dual-system-helix-groot.md) - pi-0의 단일 모델 접근과 달리, Helix(Figure AI)와 GR00T N1(NVIDIA)은 "느린 추론 + 빠른 제어"를 분리하는 이중 시스템을 제안한다. 인간의 System 1/System 2를 모방한 2025년 핵심 아키텍처 패러다임.
