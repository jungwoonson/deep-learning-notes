# Contrastive Learning / SigLIP

## VLA 연결고리

OpenVLA는 **SigLIP을 vision encoder의 두 번째 축**으로 사용한다. SigLIP은 이미지와 텍스트를 같은 공간에 정렬(align)하여, 이미지의 **semantic meaning**(의미)을 언어와 연결한다. 로봇이 "빨간 컵을 집어라"라는 명령을 이미지와 연결해 이해하려면, 이 language-aligned visual feature가 필수적이다.

---

## 핵심 개념

### 1. Contrastive Learning이란

Contrastive learning의 기본 원리:

- **Positive pair**: 서로 관련 있는 샘플 (예: 같은 이미지와 그 설명 텍스트) → 가까워지도록 학습
- **Negative pair**: 서로 관련 없는 샘플 (예: 다른 이미지와 무관한 텍스트) → 멀어지도록 학습

핵심은 **비교(contrast)**를 통해 좋은 representation을 학습하는 것이다. 정답 label이 아니라 "이것과 저것이 같은가/다른가"로 학습한다.

### 2. InfoNCE Loss

Contrastive learning의 대표적 손실 함수:

```
주어진 anchor에 대해:
  - 1개의 positive와 유사도를 높이고
  - N개의 negative와 유사도를 낮추되
  - softmax 형태로 확률 분포를 만들어 cross-entropy로 학습
```

직관적으로, N+1개 중에서 올바른 짝을 찾는 **분류 문제**로 변환하는 것이다. batch size가 클수록 더 많은 negative를 볼 수 있어 학습이 효과적이다.

### 3. CLIP의 Contrastive Pre-training

CLIP(Contrastive Language-Image Pre-training)은 contrastive learning을 image-text pair에 적용한 모델이다:

- **Image encoder** (ViT): 이미지를 벡터로 변환
- **Text encoder** (Transformer): 텍스트를 벡터로 변환
- 학습 목표: 짝이 맞는 image-text pair는 가깝게, 아닌 것은 멀게

batch 내의 모든 image-text 조합에 대해 **NxN 유사도 행렬**을 만들고, 대각선(올바른 짝)의 유사도를 최대화한다. 이것이 InfoNCE를 image-text에 적용한 것이다.

### 4. SigLIP -- Sigmoid Loss for Language-Image Pre-training

SigLIP은 CLIP의 개선 버전으로, 핵심 차이는 **손실 함수**에 있다:

| | CLIP | SigLIP |
|---|------|--------|
| Loss 방식 | Softmax (NxN 행렬 전체) | Sigmoid (각 pair 독립) |
| 계산 | 전체 batch의 유사도 필요 | 각 pair를 독립적으로 판단 |
| 스케일링 | 큰 batch에 메모리 부담 | batch 크기 확장 용이 |
| 판단 방식 | "N개 중 정답 하나 고르기" | "이 pair가 맞는가? Yes/No" |

SigLIP의 sigmoid 방식은 각 image-text pair를 **독립적인 이진 분류**(맞다/아니다)로 처리한다. 이렇게 하면:
- 더 큰 batch로 학습 가능 (메모리 효율적)
- 안정적인 학습
- 동등하거나 더 나은 성능

### 5. VLA에서 SigLIP의 역할

OpenVLA가 SigLIP을 선택한 이유:

- **Language-aligned feature**: 이미지 feature가 이미 텍스트와 정렬되어 있어, 언어 명령 이해에 유리
- **Semantic understanding**: "컵", "테이블", "집기" 같은 개념을 visual feature에 내재
- **DINOv2와의 상보성**: DINOv2가 spatial(어디에)을 담당하면, SigLIP은 semantic(무엇을, 어떤 의미로)을 담당

---

## 연습 주제

1. Contrastive learning에서 batch size가 클수록 유리한 이유를 설명해 보라
2. CLIP의 softmax 방식과 SigLIP의 sigmoid 방식의 차이를 NxN 유사도 행렬 관점에서 비교해 보라
3. "빨간 컵을 집어라"라는 명령을 처리할 때, language-aligned visual feature가 왜 필요한지 설명해 보라
4. DINOv2 (spatial) + SigLIP (semantic)의 조합이 왜 효과적인지, 각각 단독 사용 대비 장점을 추론해 보라
5. Contrastive learning과 self-supervised learning (DINO)의 학습 신호 차이를 비교해 보라

---

## 다음 노트

[Feature Extraction -- Patch Embeddings and Token Sequences](./04-feature-extraction-patches.md)
