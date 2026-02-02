# CLIP -- Contrastive Language-Image Pre-training

## VLA 연결고리

VLA는 이미지와 언어를 동시에 이해해야 한다. CLIP은 이미지와 텍스트를 **하나의 공유 공간(shared embedding space)**에 정렬하는 방법을 처음으로 대규모로 보여준 모델이다. CLIP의 원리는 SigLIP으로 발전했고, OpenVLA의 vision encoder에 직접 사용된다. CLIP은 또한 LLaVA 등 VLM(Vision-Language Model)의 핵심 구성 요소이기도 하다.

---

## 핵심 개념

### 1. 두 개의 독립된 Encoder

CLIP은 두 개의 별도 encoder로 구성된다:

- **Image Encoder**: 이미지를 고정 크기 벡터로 변환 (ViT 또는 ResNet 사용)
- **Text Encoder**: 텍스트를 같은 크기의 벡터로 변환 (Transformer 사용)

두 encoder는 각자 독립적으로 작동하지만, **출력 벡터의 차원은 동일**하다. 이를 통해 이미지와 텍스트를 같은 공간에서 비교할 수 있다.

### 2. Contrastive Loss로 학습

학습 데이터: 인터넷에서 수집한 **4억 개의 image-text pair**

학습 방식:

```
Batch 내 N개의 image-text pair에 대해:
  - N개 이미지 → Image Encoder → N개 이미지 벡터
  - N개 텍스트 → Text Encoder → N개 텍스트 벡터
  - N x N 유사도 행렬 계산 (cosine similarity)
  - 대각선 (올바른 pair): 유사도 최대화
  - 나머지 (잘못된 pair): 유사도 최소화
```

이 과정을 통해 "강아지 사진"과 "a photo of a dog"은 가까워지고, "강아지 사진"과 "a red car"는 멀어진다.

### 3. Shared Embedding Space

학습이 완료되면, 이미지와 텍스트가 **같은 벡터 공간**에 존재하게 된다:

- 의미적으로 관련된 이미지와 텍스트는 가까이 위치
- 관련 없는 것들은 멀리 위치
- 이미지끼리의 유사도, 텍스트끼리의 유사도도 자연스럽게 의미를 반영

이 공유 공간이 CLIP의 가장 강력한 특성이다.

### 4. Zero-Shot Classification

CLIP의 혁신적 능력은 **학습하지 않은 카테고리도 분류**할 수 있다는 것이다:

```
분류 과정:
1. 분류할 이미지 → Image Encoder → 이미지 벡터
2. 모든 카테고리를 텍스트로 변환 → Text Encoder → 카테고리 벡터들
   예: "a photo of a cat", "a photo of a dog", "a photo of a car"
3. 이미지 벡터와 각 카테고리 벡터의 유사도 계산
4. 가장 유사도가 높은 카테고리가 정답
```

별도의 fine-tuning 없이도 새로운 task에 적용 가능하다. 이를 **zero-shot transfer**라 한다.

### 5. CLIP의 한계와 발전

**한계:**

- 이미지 전체를 하나의 벡터로 요약 → 세밀한 공간 정보 손실
- 텍스트와 이미지의 관계를 "일치/불일치"로만 학습 → 복잡한 추론에 약함
- 이미지 생성이나 긴 텍스트 이해에는 부적합

**발전 방향:**

- **SigLIP**: softmax → sigmoid loss로 개선 (OpenVLA에서 사용)
- **LLaVA**: CLIP encoder를 LLM과 결합하여 복잡한 시각적 질문에 답변
- **OpenVLA**: CLIP/SigLIP의 semantic feature를 로봇 행동 결정에 활용

---

## 연습 주제

1. CLIP이 "shared embedding space"를 만드는 과정을 자신의 말로 설명해 보라
2. Zero-shot classification이 기존 supervised classification과 어떻게 다른지 비교해 보라
3. CLIP의 contrastive loss에서 batch size가 왜 중요한 역할을 하는지 설명해 보라
4. 이미지 전체를 하나의 벡터로 요약하는 것이 로봇 task에서 왜 한계가 되는지 생각해 보라
5. CLIP → SigLIP → VLA로 이어지는 기술 발전의 흐름을 정리해 보라

---
