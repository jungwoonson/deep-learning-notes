# LLaVA -- Large Language and Vision Assistant

## VLA 연결고리

LLaVA는 **OpenVLA의 직접적 조상**이다. "Vision Encoder + Projector + LLM"이라는 3단 구조를 확립했고, visual instruction tuning이라는 학습 방법을 제시했다. OpenVLA는 이 구조를 거의 그대로 계승하되, 출력을 텍스트에서 로봇 행동으로 바꾼 것이다.

---

## 핵심 개념

### 1. LLaVA의 구조

LLaVA는 세 가지 구성 요소로 이루어진다:

```
이미지 → CLIP Vision Encoder (ViT-L/14) → visual feature
              ↓
        Linear Projector → visual token (LLM 차원에 맞춤)
              ↓
  [visual tokens] + [text tokens] → Llama LLM → 텍스트 답변 생성
```

**각 구성 요소의 역할:**

- **CLIP Vision Encoder**: 이미지를 의미 있는 visual feature로 변환. 사전 학습된 가중치를 그대로 사용 (frozen)
- **Linear Projector**: vision feature를 LLM이 이해할 수 있는 차원으로 변환하는 다리(bridge) 역할
- **Llama LLM**: visual token과 text token을 함께 받아 자연어 답변을 생성

### 2. 핵심 통찰 -- 단순함의 힘

LLaVA의 가장 놀라운 점은 **단순한 linear layer 하나**로 vision과 language를 연결했다는 것이다. 복잡한 fusion 모듈 없이도, 잘 학습된 vision encoder와 LLM 사이에 간단한 projector만 두면 놀라운 멀티모달 능력이 나타났다.

이 통찰이 이후 VLM/VLA 연구의 방향을 결정했다: **기존 강력한 모델을 잘 연결하는 것**이 핵심이다.

### 3. Visual Instruction Tuning

LLaVA의 학습 방법은 **두 단계**로 나뉜다:

**Stage 1 -- Pre-training (Feature Alignment)**

- 목적: Projector가 vision feature를 LLM 공간에 정렬하도록 학습
- 데이터: 간단한 image-caption pair (예: "이 이미지를 설명해라" → caption)
- 학습 대상: **Projector만** 학습 (Vision Encoder와 LLM은 frozen)
- 비유: 통역사가 두 언어의 기본 대응 관계를 배우는 단계

**Stage 2 -- Fine-tuning (Visual Instruction Tuning)**

- 목적: 다양한 시각적 질문과 지시에 따를 수 있도록 전체 모델을 조정
- 데이터: GPT-4로 생성한 다양한 visual instruction 데이터
  - 상세 설명, 복잡한 추론, 대화 등
- 학습 대상: **Projector + LLM** 함께 학습 (Vision Encoder는 보통 frozen)
- 비유: 통역사가 다양한 실전 상황에서 훈련하는 단계

### 4. Instruction Tuning 데이터의 중요성

LLaVA는 GPT-4를 활용하여 고품질 instruction 데이터를 생성했다:

- **Detailed description**: "이 이미지를 자세히 설명해라"
- **Complex reasoning**: "이 장면에서 일어나고 있는 일을 추론해라"
- **Conversation**: 이미지에 대한 다단계 대화

이 데이터의 질이 모델 성능을 크게 좌우한다. 단순한 caption보다 다양하고 복잡한 instruction이 모델의 범용 능력을 키운다.

### 5. LLaVA에서 OpenVLA로

LLaVA와 OpenVLA의 관계:

| | LLaVA | OpenVLA |
|---|-------|---------|
| Vision Encoder | CLIP ViT-L/14 | DINOv2 + SigLIP (dual) |
| Projector | Linear layer | MLP |
| LLM | Llama | Llama 2 |
| 출력 | 텍스트 답변 | 로봇 행동 (action token) |
| 학습 데이터 | Visual instruction | 로봇 조작 demonstration |

구조는 거의 동일하지만, **vision encoder가 더 강화**되고 (dual encoder), **출력이 행동**으로 바뀐 것이 핵심 차이다. LLaVA → Prismatic VLM → OpenVLA 순으로 발전했다.

---

## 연습 주제

1. LLaVA의 2단계 학습(pre-training → fine-tuning)에서 각 단계가 왜 필요한지 설명해 보라
2. Projector가 "단순한 linear layer"인데도 효과적인 이유를 추론해 보라
3. Stage 1에서 LLM을 frozen하고 projector만 학습하는 이유를 설명해 보라
4. Visual instruction tuning 데이터의 다양성이 왜 중요한지, 단순 caption만으로는 부족한 이유를 생각해 보라
5. LLaVA의 구조를 로봇에 적용하려면 어떤 변경이 필요할지 자신만의 아이디어를 정리해 보라

---

## 다음 노트

[Prismatic VLM -- OpenVLA의 기반 모델](./04-prismatic-vlm.md)
