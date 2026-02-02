# 고전 네트워크와 ResNet (Classic Networks and ResNet)

## 왜 필요한가

VLA 모델의 vision encoder(예: ViT, ResNet backbone)는 수십 년간 축적된 CNN 아키텍처 발전의 결과물이다. 특히 **ResNet의 skip connection** 개념은 Transformer에서 그대로 재사용되며, VLA 모델의 핵심 구성 요소다. 고전 네트워크의 발전 과정을 이해하면 "왜 현대 아키텍처가 이렇게 생겼는가"에 대한 답을 얻을 수 있다.

---

## 핵심 개념

### 1. LeNet-5 (1998)

CNN의 원조이자, 현대 CNN 구조의 청사진이다.

- **설계 목적**: 손글씨 숫자 인식 (MNIST, 32x32 흑백 이미지)
- **구조**: Conv → Pool → Conv → Pool → FC → FC → 출력
- **핵심 아이디어**: 합성곱으로 공간적 패턴을 추출하고 FC로 분류하는 이 기본 흐름은 지금도 유효하다.
- **역사적 의의**: "학습 가능한 필터가 수작업 특징 추출보다 낫다"를 증명했다.
- 현대 기준으로 매우 작은 네트워크 (약 6만 파라미터).

### 2. AlexNet (2012)

딥러닝 혁명의 시작점이다.

- **설계 목적**: ImageNet 대규모 이미지 분류 (224x224 컬러, 1000 클래스)
- **핵심 혁신**:
  - GPU를 사용한 대규모 학습
  - ReLU 활성화 함수 도입 (sigmoid/tanh 대비 학습 속도 대폭 향상)
  - Dropout으로 과적합 방지
  - Data augmentation 활용
- **구조**: 5개 Conv 층 + 3개 FC 층, 약 6천만 파라미터
- **의의**: 2012년 ImageNet 대회에서 압도적 성능으로 우승하며, 컴퓨터 비전 패러다임을 전통적 방법에서 딥러닝으로 전환시켰다.

### 3. VGGNet (2014)

"단순함의 힘"을 보여준 네트워크다.

- **핵심 아이디어**: 오직 **3x3 커널**만 사용하되, 층을 매우 깊게 쌓는다.
- **구조 패턴**: 3x3 Conv 여러 개 → MaxPool → 반복 (VGG-16: 16층, VGG-19: 19층)
- **왜 3x3인가**:
  - 3x3 두 층 = 5x5 한 층의 receptive field (하지만 파라미터가 적고 비선형성이 추가됨)
  - 3x3 세 층 = 7x7 한 층의 receptive field
- **약점**: FC 층의 파라미터가 약 1억 2천만 개로 매우 무겁다.
- **의의**: "깊이가 성능에 중요하다"는 교훈을 확립했다.

### 4. Degradation Problem (성능 저하 문제)

네트워크를 계속 깊게 만들면 성능이 계속 좋아질까?

- **기대**: 층이 많을수록 더 복잡한 함수를 학습할 수 있으므로 성능이 좋아야 한다.
- **현실**: 일정 깊이를 넘으면 **학습 오차(training error)마저 증가**한다.
- 이것은 과적합이 아니다. 과적합이면 training error는 낮고 test error만 높아야 한다.
- 원인: 매우 깊은 네트워크에서 gradient가 효과적으로 전파되지 못하고, 최적화 자체가 어려워진다.
- 이론적으로, 추가 층이 **항등 함수(identity function)** 를 학습하면 최소한 얕은 네트워크만큼의 성능은 보장되어야 하지만, 실제로는 항등 함수조차 학습하기 어렵다.

### 5. ResNet (2015) -- Skip/Residual Connection

Degradation problem의 해결책이자, 현대 딥러닝에서 가장 영향력 있는 아이디어 중 하나다.

- **핵심 아이디어**: 층이 입력을 직접 변환하는 대신 **잔차(residual)** 만 학습하게 한다.

```
일반 네트워크:  출력 = F(x)           ← x에서 출력을 직접 학습
ResNet:        출력 = F(x) + x       ← x에서의 "변화량"만 학습
```

- **Skip connection**(또는 shortcut connection): 입력 x를 층의 출력에 직접 더해준다.
- **왜 효과적인가**:
  - 추가 층이 아무것도 배울 것이 없으면 F(x) = 0을 학습하면 된다 → 항등 함수가 쉽게 구현됨
  - Gradient가 skip connection을 통해 직접 앞쪽 층으로 흐를 수 있다 → 깊은 네트워크 학습 가능
- **구조**: Residual Block을 수십~수백 개 쌓는다 (ResNet-50, ResNet-101, ResNet-152)
- **성과**: 152층 네트워크가 안정적으로 학습되어 ImageNet 우승

### 6. Skip Connection이 Transformer에서 재등장하는 이유

이것이 VLA 학습 경로에서 ResNet을 반드시 이해해야 하는 핵심 이유다.

- Transformer의 모든 블록은 **Multi-Head Attention + Skip Connection** 과 **Feed-Forward + Skip Connection** 으로 구성된다.
- 수식 구조가 ResNet과 동일하다: `출력 = Layer(x) + x`
- Skip connection 없이는 수십 층의 Transformer를 학습할 수 없다.
- VLA 모델 내부의 vision encoder(ViT)와 language model(Transformer) 모두 이 구조를 사용한다.
- **ResNet에서 이 개념을 확실히 이해하면, Transformer 학습이 훨씬 수월해진다.**

---

## 발전 흐름 요약

| 네트워크 | 연도 | 깊이 | 핵심 기여 |
|---------|------|------|----------|
| LeNet-5 | 1998 | 5층 | CNN의 기본 구조 확립 |
| AlexNet | 2012 | 8층 | GPU + ReLU + Dropout으로 딥러닝 부활 |
| VGG | 2014 | 16-19층 | 3x3 커널 반복, 깊이의 중요성 |
| ResNet | 2015 | 50-152층 | Skip connection으로 초깊은 네트워크 학습 |

---

## 연습 주제

1. VGG-16의 각 블록별 출력 크기와 파라미터 수를 추적해 보라 (입력 224x224x3).
2. 3x3 Conv 두 층의 receptive field가 5x5 Conv 한 층과 같음을 그림으로 확인하고, 파라미터 수를 비교해 보라.
3. Residual Block에서 skip connection이 있을 때와 없을 때, gradient 흐름의 차이를 직관적으로 설명해 보라.
4. ResNet에서 입력 x와 F(x)의 차원이 다를 때(예: 채널 수가 변할 때) 어떻게 처리하는지 조사해 보라.
5. ResNet-50의 bottleneck block이 1x1 Conv를 어떻게 활용하는지 구조를 그려보라.

---
