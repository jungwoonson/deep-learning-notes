# Vision Transformer (ViT)

## VLA 연결고리

VLA(Vision-Language-Action) 모델은 이미지를 이해해야 로봇 행동을 결정할 수 있다. ViT는 이미지를 **token sequence**로 변환하는 핵심 구조로, VLA의 vision encoder 내부에서 이 방식이 그대로 사용된다. OpenVLA가 사용하는 DINOv2와 SigLIP 모두 ViT 아키텍처 기반이다.

---

## 핵심 개념

### 1. CNN에서 Transformer로

기존 컴퓨터 비전은 CNN(Convolutional Neural Network)이 지배했다. CNN은 local feature를 잘 잡지만, 이미지 전체의 global relationship을 파악하는 데 한계가 있었다. NLP에서 큰 성공을 거둔 Transformer를 vision에 적용하자는 아이디어가 ViT의 출발점이다.

핵심 질문: **이미지를 어떻게 Transformer에 넣을 수 있을까?**

### 2. Patch Embedding

ViT의 핵심 아이디어는 이미지를 작은 **patch**(조각)로 나누는 것이다.

- 예: 224x224 이미지를 16x16 크기의 patch로 나누면 14x14 = **196개의 patch**가 생긴다
- 각 patch를 linear projection으로 고정 크기 벡터(embedding)로 변환한다
- 이렇게 하면 이미지가 **196개의 token sequence**가 된다 -- NLP에서 문장을 단어 token으로 나누는 것과 동일한 원리

### 3. ViT 아키텍처

전체 흐름은 다음과 같다:

```
이미지 → patch로 분할 → patch embedding → position embedding 추가
→ [CLS] token 추가 → Transformer Encoder (반복) → classification head
```

**구성 요소:**

- **Patch Embedding**: 각 patch를 벡터로 변환
- **Position Embedding**: 각 patch의 위치 정보를 더해줌 (Transformer는 순서를 모르므로)
- **CLS Token**: sequence 맨 앞에 추가하는 특수 token. 이미지 전체를 대표하는 요약 역할
- **Transformer Encoder**: Multi-Head Self-Attention + Feed-Forward Network의 반복
- **Classification Head**: CLS token의 최종 출력으로 분류 수행

### 4. CLS Token의 역할

- BERT에서 빌려온 개념이다
- 모든 patch token과 attention을 주고받으며 이미지 전체 정보를 집약한다
- 최종 분류는 이 CLS token 하나의 출력만으로 수행한다
- VLA에서는 CLS token 대신 **모든 patch token의 출력**을 사용하는 경우가 많다 (공간 정보 보존을 위해)

### 5. ViT Variants

| 모델 | 파라미터 수 | patch 크기 | 특징 |
|------|-----------|-----------|------|
| ViT-Base | ~86M | 16x16 | 기본 모델 |
| ViT-Large | ~307M | 16x16 | 더 큰 모델 |
| ViT-Huge | ~632M | 14x14 | 최대 모델 |

patch 크기가 작을수록 token 수가 많아져 세밀한 정보를 담지만, 계산 비용이 크게 증가한다.

---

## 왜 중요한가

ViT가 보여준 "이미지를 patch token sequence로 변환" 패러다임은 이후 모든 vision 모델의 기반이 되었다. DINO, SigLIP, CLIP 모두 ViT 위에 세워졌고, VLA도 이 구조를 통해 이미지를 처리한다.

---

## 연습 주제

1. 384x384 이미지를 16x16 patch로 나누면 token이 몇 개 생기는지 계산해 보라
2. Position embedding이 없으면 어떤 문제가 생기는지 설명해 보라
3. CNN의 convolution과 ViT의 self-attention이 "이미지 내 관계 파악"에서 어떻게 다른지 비교해 보라
4. CLS token 방식과 모든 patch token을 사용하는 방식의 장단점을 정리해 보라
5. patch 크기를 줄이면 (예: 16x16 → 8x8) 성능과 비용이 어떻게 변하는지 추론해 보라

---

## 다음 노트

[DINO / DINOv2 -- Self-Supervised Vision Learning](./02-dino-dinov2.md)
