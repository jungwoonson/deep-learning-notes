# VLA 개요 2026 (Vision-Language-Action Overview)

## VLA와의 연결

**이 노트는 VLA 시리즈의 첫 번째이자 전체 커리큘럼의 최종 파트이다.** 지금까지 배운 모든 것 -- 선형대수, 딥러닝, CNN, Transformer, LLM, 멀티모달, 강화학습, Diffusion -- 이 VLA라는 하나의 목표를 향해 수렴한다. VLA는 로봇이 **보고(Vision)**, **이해하고(Language)**, **행동하는(Action)** 통합 시스템이다. 2024년 이후 로봇 AI의 가장 중요한 패러다임이며, 2026년 현재 가장 활발히 연구되는 분야이다.

---

## 핵심 개념

### 1. VLA란 무엇인가

VLA(Vision-Language-Action Model)는 세 가지 modality를 하나의 모델에서 처리한다:

```
Vision (시각)       Language (언어)         Action (행동)
카메라 이미지  +    자연어 명령        →    로봇 행동 출력
  ↓                    ↓                      ↓
ViT, DINOv2       Llama, PaLM          관절 각도, 속도
SigLIP             텍스트 토큰           그리퍼 열기/닫기
```

핵심 아이디어: **사전학습된 Vision-Language Model(VLM)을 로봇 행동 데이터로 파인튜닝하여 로봇을 제어한다.**

이것이 혁명적인 이유:
- 기존 로봇 제어: 각 task마다 별도 알고리즘, 제한된 환경만 동작
- VLA 접근: 인터넷 규모의 시각+언어 지식을 로봇에 이전 (transfer)
- 결과: "빨간 컵을 집어서 파란 접시 위에 놓아라" 같은 자연어 명령을 처리 가능

### 2. VLA 패러다임의 핵심 통찰

```
전통적 로봇:
  센서 → 인지 모듈 → 계획 모듈 → 제어 모듈 → 모터
  (각 모듈을 별도 설계, 수작업 규칙 기반)

VLA 패러다임:
  카메라 이미지 + 언어 명령 → [단일 신경망] → 행동 출력
  (End-to-End, 데이터에서 학습)
```

VLA의 핵심 통찰 세 가지:

1. **인터넷 사전학습의 힘**: VLM이 인터넷에서 수십억 장의 이미지와 텍스트를 학습하면서 얻은 "세상에 대한 이해"를 로봇에 전달할 수 있다
2. **행동을 토큰으로**: 로봇의 행동(관절 각도, 그리퍼 상태)을 언어 모델의 토큰으로 표현하면, 텍스트 생성과 같은 방식으로 행동을 생성할 수 있다
3. **범용 로봇**: 하나의 모델로 다양한 로봇, 다양한 task를 수행 → 범용 로봇 에이전트(generalist robot agent)

### 3. 역사 타임라인

```
2022.12  RT-1 (Google DeepMind)
         최초의 대규모 로봇 Transformer. 13만 에피소드로 학습.
         EfficientNet + Transformer. 로봇 전용 데이터만 사용.
         한계: 학습에 없는 물체나 환경에 취약 (poor generalization)

2023.07  RT-2 (Google DeepMind)
         ★ 패러다임 전환 ★ 사전학습된 VLM을 로봇 데이터로 파인튜닝.
         행동을 텍스트 토큰으로 표현한 최초의 모델.
         VLM의 지식 덕분에 학습에 없는 물체도 조작 가능.

2024.06  OpenVLA (Stanford/Berkeley/TRI)
         최초의 오픈소스 VLA. 7B 파라미터.
         DINOv2 + SigLIP → Llama 2 7B.
         Open X-Embodiment 데이터셋으로 학습.

2024.10  pi-0 (Physical Intelligence)
         Flow Matching 기반 VLA. 50Hz 제어.
         7개 로봇 플랫폼에서 검증. 연속적 행동 생성.

2025.02  Helix (Figure AI)
         Dual-System 아키텍처의 시작.
         System 2 (7B VLM, 추론) + System 1 (80M, 200Hz 제어).

2025.03  GR00T N1 (NVIDIA)
         Eagle-2 VLM + Diffusion Transformer.
         같은 Dual-System 패턴. 로봇 범용 파운데이션 모델 표방.

2025.04  SmolVLA (Hugging Face)
         450M 파라미터. 소비자 GPU에서 실행 가능.
         VLA 민주화의 시작. LeRobot 커뮤니티 데이터셋 활용.

2025.06  pi-0-FAST / pi-0.5 (Physical Intelligence)
         FAST 토크나이저로 5배 빠른 학습.
         pi-0.5는 open-world generalization 달성.

2025.08  pi-star-0.6 (Physical Intelligence)
         RECAP: 로봇 경험에서의 RL.
         18시간 자율 운영. 자기 개선하는 VLA.

2025.11  GEN-0 (Genesis Embodied AI)
         물리 시뮬레이션 통합 VLA. sim-to-real 전이.

2026.01  GR00T N1.6 (NVIDIA, CES 2026)
         N1의 발전형. 더 빠른 추론, 더 넓은 로봇 지원.
```

### 4. VLA Taxonomy (분류 체계)

VLA 모델을 분류하는 주요 기준:

```
[행동 표현 방식]
├── 이산 토큰 (Discrete Tokens)
│   ├── RT-2: 행동을 텍스트 토큰으로
│   └── OpenVLA: 256 bins per dimension
├── 압축 토큰 (Compressed Tokens)
│   ├── FAST: DCT 기반 압축
│   └── FASTer: 더 효율적인 압축
└── 연속 분포 (Continuous Distribution)
    ├── pi-0: Flow Matching
    └── SmolVLA: Flow-Matching Transformer

[아키텍처 패턴]
├── 단일 모델 (Monolithic)
│   ├── RT-2, OpenVLA, pi-0
│   └── 하나의 모델이 인지+행동 전부 처리
└── 이중 시스템 (Dual-System)
    ├── Helix: System 1 (빠른 제어) + System 2 (느린 추론)
    └── GR00T N1: 같은 패턴, 다른 구현

[모델 규모]
├── 대형 (7B+): RT-2 (55B), OpenVLA (7B), pi-0 (3B VLM+)
├── 중형 (1-3B): GR00T N1
└── 소형 (<1B): SmolVLA (450M)

[데이터 소스]
├── 폐쇄적: RT-1 (Google 내부 로봇), pi-0 (PI 자체 수집)
└── 개방적: OpenVLA (Open X-Embodiment), SmolVLA (LeRobot)
```

### 5. 로봇 산업의 폭발적 성장

2025년은 로봇 AI 투자의 해였다:

```
2025년 로보틱스 VC 투자: 약 $7.2B (역대 최대)

주요 투자:
  Physical Intelligence:   $600M (Series A, 역대 로보틱스 최대 라운드)
  Figure AI:               $675M (Series B)
  NVIDIA GR00T 에코시스템:  Jetson Thor + Isaac Sim + GR00T 모델
  Hugging Face LeRobot:    오픈소스 로봇 학습 프레임워크

왜 지금인가?
  1. LLM/VLM 성능이 로봇에 전이 가능한 수준 도달
  2. 하드웨어 가격 하락 (저가 로봇 팔: $10K 이하)
  3. 데이터 수집 인프라 성숙 (텔레오퍼레이션, 시뮬레이션)
  4. Open X-Embodiment 등 공유 데이터셋 등장
```

### 6. 이 시리즈에서 배울 것

```
Note 01: [현재] VLA 개요 2026 -- 전체 지도
Note 02: RT-1/RT-2 -- VLA의 기원
Note 03: OpenVLA 아키텍처 -- 오픈소스 VLA의 내부 구조
Note 04: pi-zero 계열 -- Flow Matching 기반 VLA
Note 05: Dual-System (Helix/GR00T) -- 2025년 핵심 패러다임
Note 06: SmolVLA -- VLA 민주화
Note 07: Action Representations -- 행동 표현 방법론
Note 08: Datasets & Open X-Embodiment -- 데이터의 힘
Note 09: LeRobot으로 파인튜닝 -- 실습 파이프라인
Note 10: Fairino FR3 배포 -- 실제 로봇에 VLA 탑재
```

### 7. 이전 커리큘럼과의 연결

지금까지 배운 것이 VLA에서 어떻게 사용되는지:

```
수학 기초     → Attention 계산, Loss Function, Gradient Descent
Python        → 모든 구현의 기반
NumPy         → 데이터 전처리, 텐서 연산
ML 기초       → 분류, 회귀, 과적합, 평가 지표
PyTorch       → VLA 학습/추론 프레임워크
Neural Nets   → MLP Projector, 활성화 함수, 정규화
CNN           → DINOv2, SigLIP의 기반 아키텍처 (ViT)
시퀀스 모델    → 자기회귀 생성, 토큰 시퀀스
Attention     → Self-Attention, Cross-Attention, Multi-Head
LLM           → Llama 2 백본, 토크나이저, LoRA 파인튜닝
Modern Vision → ViT, DINOv2, SigLIP, CLIP
멀티모달       → VLM, 이미지-텍스트 정렬, Projector
RL/Imitation  → 로봇 정책 학습, 행동 복제, 보상 설계
Diffusion/Flow → Flow Matching (pi-0, SmolVLA), DDPM
```

---

## 연습 주제 (코드 없이 생각해보기)

1. **VLA 구성 요소 매핑**: VLA의 Vision, Language, Action 세 부분이 각각 이 커리큘럼의 어떤 파트에 해당하는지 정리하라. 예를 들어, Vision 부분은 CNN/Modern Vision에서 배운 어떤 모델을 사용하는가?

2. **RT-2의 혁신 이해**: RT-1(로봇 전용 모델)에서 RT-2(VLM 파인튜닝)로의 전환이 왜 혁명적인지 설명하라. "인터넷 지식의 전이"가 구체적으로 무엇을 의미하는지 예를 들어 설명하라.

3. **행동 표현 비교**: 로봇 팔의 7-DoF(자유도) 행동을 "이산 토큰"으로 표현하는 것과 "연속 분포"로 표현하는 것의 차이점을 직관적으로 설명하라. 각각의 장단점은 무엇일까?

4. **Dual-System의 직관**: 인간의 뇌도 "빠른 반사"와 "느린 사고"를 구분한다 (Daniel Kahneman의 System 1/System 2). 로봇에서 200Hz 제어(System 1)와 7Hz 추론(System 2)이 왜 분리되어야 하는지, 인간의 예시를 들어 설명하라.

5. **규모와 접근성**: 55B 파라미터의 RT-2와 450M 파라미터의 SmolVLA는 약 120배 차이가 난다. 왜 작은 모델도 중요한가? 연구/산업/교육 관점에서 각각 설명하라.

6. **$7.2B 투자의 의미**: 2025년 로보틱스 VC가 폭발한 이유를 기술적 관점(VLM 성숙, 데이터셋, 하드웨어)에서 정리하라. 이전에는 왜 불가능했는가?

---

## 다음 노트

**다음**: [RT-1/RT-2](./02-rt1-rt2.md) - VLA의 기원. EfficientNet 기반의 RT-1이 왜 한계를 보였고, RT-2가 "사전학습 VLM + 로봇 데이터 파인튜닝"이라는 핵심 아이디어로 어떻게 패러다임을 바꿨는지.
