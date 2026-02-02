# 활성화 함수 (Activation Functions)

## 왜 알아야 하는가 (Why This Matters for VLA)

활성화 함수(Activation Function)는 신경망에 **비선형성(nonlinearity)**을 부여하는 핵심 요소이다. 활성화 함수가 없으면 100층을 쌓아도 결국 하나의 거대한 선형 변환(행렬곱)에 불과하다.

VLA와의 연결 고리:
- VLA 내부의 Transformer는 활성화 함수로 **GELU** 또는 **SiLU(Swish)**를 사용한다
  - GELU: GPT, BERT, ViT 등 대부분의 Transformer 계열
  - SiLU/Swish: Llama 2, Llama 3 등 최신 LLM과 이를 기반으로 한 VLA(OpenVLA, pi-zero 등)
- 활성화 함수의 선택이 **학습 안정성과 성능**에 직접적으로 영향을 미친다
- 기울기 소실(vanishing gradient) 문제를 이해하면, 왜 깊은 모델에서 특정 활성화 함수가 선호되는지 알 수 있다
- 활성화 함수의 역사를 따라가면 딥러닝 발전사 자체를 이해할 수 있다

---

## 핵심 개념 (Core Concepts)

### 1. 왜 활성화 함수가 필요한가 (Why Nonlinearity)

활성화 함수가 없는 2층 네트워크를 생각해보자:

활성화 함수 없이:

$$\begin{aligned}
z_1 &= W_1 x + b_1 \\
z_2 &= W_2 z_1 + b_2 \\
    &= W_2 (W_1 x + b_1) + b_2 \\
    &= (W_2 W_1) x + (W_2 b_1 + b_2) \\
    &= W_{\text{combined}} \, x + b_{\text{combined}}
\end{aligned}$$

결론: 아무리 층을 쌓아도 결국 하나의 선형 변환! $W_{\text{combined}} = W_2 W_1$ 이라는 단일 행렬로 압축된다.

활성화 함수가 있으면:

$$\begin{aligned}
z_1 &= W_1 x + b_1 \\
a_1 &= f(z_1) \quad \leftarrow \text{비선형 함수 적용!} \\
z_2 &= W_2 a_1 + b_2 \\
a_2 &= f(z_2)
\end{aligned}$$

$f()$ 때문에 $W_2 W_1$으로 합칠 수 없다.
→ 각 층이 독립적인 의미를 가진다.
→ 깊이를 쌓을수록 더 복잡한 함수를 표현할 수 있다.

### 2. Sigmoid (시그모이드)

출력을 **0과 1 사이**로 압축한다. 가장 오래된 활성화 함수 중 하나.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
그래프:
  1.0 |          ___________
      |         /
  0.5 |--------/------------ (z=0일 때 0.5)
      |       /
  0.0 |______/
      +----|----|----|----→ z
         -4    0    4
```

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \quad \text{(미분 최댓값: 0.25, } z=0 \text{일 때)}$$

**특징**:
- 출력 범위: (0, 1) -- 확률로 해석 가능
- 용도: 이진 분류의 **출력층**, 게이트 메커니즘(LSTM, GRU)
- **현대 딥러닝에서 은닉층에는 거의 사용하지 않는다**

**문제점**:

1. **기울기 소실 (Vanishing Gradient)**: 미분 최댓값이 $0.25$ → 층이 깊어지면 $0.25^n$으로 기울기가 급격히 줄어듦
2. **출력이 0 중심이 아님 (not zero-centered)**: 출력이 항상 양수(0~1) → 다음 층 가중치의 gradient 방향이 편향됨 → 학습이 지그재그로 느리게 진행
3. $\exp()$ 연산이 상대적으로 비쌈

### 3. Tanh (하이퍼볼릭 탄젠트)

출력을 **-1과 1 사이**로 압축한다. Sigmoid의 개선 버전.

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = 2\sigma(2z) - 1$$

```
그래프:
  1.0 |          ___________
      |         /
  0.0 |--------/------------ (z=0일 때 0)
      |       /
 -1.0 |______/
      +----|----|----|----→ z
         -4    0    4
```

$$\tanh'(z) = 1 - \tanh^2(z) \quad \text{(미분 최댓값: 1.0, } z=0 \text{일 때)}$$

**Sigmoid 대비 개선**:
- **0 중심** 출력 → gradient 방향 편향 문제 완화
- 미분 최댓값이 1.0으로 더 큼 → 기울기 소실이 sigmoid보다 덜함
- LSTM/GRU의 내부 상태 계산에 여전히 사용됨

**남아있는 문제**: 여전히 $|z|$가 크면 미분이 0에 가까워짐 (포화 문제, saturation)

### 4. ReLU (Rectified Linear Unit)

딥러닝의 **기본(default)** 활성화 함수. 2012년 AlexNet 이후 표준이 되었다.

$$\text{ReLU}(z) = \max(0, z)$$

```
그래프:
      |       /
      |      /
      |     /
      |    /
  0   |___/
      +----|----|----|----→ z
         -4    0    4
```

$$\text{ReLU}'(z) = \begin{cases} 0 & z < 0 \\ 1 & z > 0 \\ \text{정의 안 됨} & z = 0 \text{ (실무에서는 0 또는 1)} \end{cases}$$

**왜 ReLU가 혁명적이었나**:

Sigmoid/Tanh의 문제:
- 미분 최댓값이 $0.25$ / $1.0$ → 층이 깊으면 기울기 소실
- $\exp()$ 연산이 비쌈

ReLU의 장점:
1. $z > 0$일 때 미분 $= 1$ → 기울기가 소실되지 않고 그대로 전달
2. $\max()$ 연산만 필요 → 매우 빠름
3. 희소 활성화: $z < 0$인 뉴런은 출력 $0$ → 계산 효율적
4. 깊은 네트워크 학습을 실질적으로 가능하게 함

**ReLU의 문제 -- Dead Neuron (죽은 뉴런)**:
```
z가 항상 음수인 뉴런:
  출력 = 항상 0
  기울기 = 항상 0
  → 가중치 업데이트 불가 → 영구히 "죽음"

원인: 학습률이 너무 크면 가중치가 크게 변하여
      특정 뉴런의 입력이 항상 음수가 될 수 있다

해결: LeakyReLU, 적절한 학습률, 적절한 가중치 초기화
```

### 5. LeakyReLU

ReLU의 Dead Neuron 문제를 해결한 변형.

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases} \quad (\alpha \text{는 보통 } 0.01 \text{ 또는 } 0.1)$$

```
그래프 (alpha=0.1):
      |       /
      |      /
      |     /
      |    /
  0   |   / ← 기울기 1
      |__/ ← 기울기 0.1 (완전한 0이 아님!)
      +----|----|----|----→ z
         -4    0    4
```

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$$

**핵심**: $z < 0$일 때도 작은 기울기가 존재하므로 뉴런이 "죽지" 않는다.

### 6. GELU (Gaussian Error Linear Unit) -- Transformer의 표준

$$\text{GELU}(z) = z \cdot \Phi(z) \quad (\Phi(z) = \text{정규분포의 누적분포함수 (CDF)})$$

근사식:

$$\text{GELU}(z) \approx 0.5 \cdot z \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 \, z^3\right)\right)\right)$$

```
그래프:
      |       /  ← z가 크면 ReLU와 거의 같음
      |      /
      |     /
      |    /
  0   |  _/ ← z=0 근처에서 매끄러운 곡선
      | / ← z가 작은 음수일 때 살짝 음의 값
  -0.2|/
      +----|----|----|----→ z
         -4    0    4
```

**GELU가 특별한 이유**:

- **ReLU**: $z > 0$이면 무조건 통과, $z < 0$이면 무조건 차단 (결정적)
- **GELU**: $z$의 값에 따라 "확률적으로" 통과시킴 (부드러운 게이팅) → $z$가 클수록 높은 확률로 통과 → $z$가 작을수록 낮은 확률로 통과 (차단이 아니라 "감쇠")

이 "부드러운 게이팅"이 Transformer에서 더 좋은 성능을 보인다.

**사용되는 곳**:
- BERT, GPT-2, GPT-3, ViT
- 대부분의 Transformer 기반 모델의 FFN

### 7. SiLU / Swish -- Llama 2와 VLA의 선택

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

$$\text{Swish}(z) = z \cdot \sigma(\beta z) \quad (\beta = 1 \text{일 때 SiLU와 동일})$$

```
그래프:
      |       /  ← z가 크면 거의 z 자체
      |      /
      |     /
      |    /
  0   |  _/
      | / ← z가 음수일 때 약간 음의 값 후 0으로 수렴
 -0.3|/
      +----|----|----|----→ z
         -4    0    4
```

**GELU와의 비교**:

유사점:
- 둘 다 매끄러운 곡선 (미분 가능)
- 둘 다 $z < 0$에서 완전히 $0$이 아닌 작은 음의 값 허용
- 둘 다 "부드러운 게이팅" 효과

차이점:
- SiLU $= z \cdot \sigma(z)$, GELU $= z \cdot \Phi(z)$
- SiLU가 계산이 약간 더 단순
- SiLU의 음의 최솟값이 약 $-0.28$, GELU는 약 $-0.17$
- 실무에서 성능 차이는 미미하나, 모델 설계 시 하나를 선택해야 함

**사용되는 곳**:
- **Llama 2, Llama 3** (SwiGLU 변형: SiLU와 Gated Linear Unit의 결합)
- **OpenVLA, pi-zero** 등 Llama 기반 VLA 모델
- EfficientNet, 일부 최신 비전 모델

### 8. 기울기 소실 문제 총정리 (Vanishing Gradient Problem)

깊은 네트워크에서 역전파 시 기울기가 **0에 수렴하여 앞쪽 층이 학습되지 않는** 문제.

역전파 시 체인룰:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial z_n} \cdot \frac{\partial z_n}{\partial a_{n-1}} \cdots \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

각 $\frac{\partial a_k}{\partial z_k}$ = 활성화 함수의 미분

- **Sigmoid**: 미분 최댓값 $0.25$ → 10층: $0.25^{10} \approx 0.000001$ → 기울기가 사실상 소멸. 앞쪽 층은 학습이 거의 일어나지 않음
- **ReLU**: 미분값 $0$ 또는 $1$ → $z > 0$인 경로: $1^{10} = 1$ → 기울기가 온전히 전달. $z < 0$인 경로: $0$ → 해당 뉴런 비활성화 (but 전체가 죽지는 않음)
- **GELU/SiLU**: 미분값이 $0$과 $1$ 사이에서 부드럽게 변함. 기울기 소실 위험이 ReLU보다 약간 높지만, Transformer에서는 잔차 연결(Residual Connection)이 이 문제를 해결

**기울기 소실을 해결하는 현대적 방법들**:
```
1. 활성화 함수 선택:  ReLU 계열 (ReLU, LeakyReLU, GELU, SiLU)
2. 잔차 연결:        ResNet, Transformer의 skip connection
3. 적절한 초기화:     Xavier, Kaiming 초기화
4. 정규화 기법:       BatchNorm, LayerNorm, RMSNorm
5. 기울기 클리핑:     gradient clipping

→ Transformer/VLA는 2~4를 모두 사용하여 100층 이상도 안정적으로 학습
```

### 9. 활성화 함수 선택 가이드 (Practical Guide)

```
상황                        → 추천 활성화 함수
일반적인 MLP/CNN 은닉층      → ReLU (기본값) 또는 LeakyReLU
Transformer FFN             → GELU (BERT, GPT, ViT 계열)
                            → SiLU/SwiGLU (Llama 계열, VLA)
이진 분류 출력층              → Sigmoid
다중 분류 출력층              → Softmax (활성화 함수라기보다 정규화)
회귀 출력층                  → 없음 (Linear/Identity)
RNN/LSTM 내부               → Tanh, Sigmoid (게이트)
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **비선형성 필요성**: 활성화 함수 없이 3층 MLP를 수식으로 전개하여, 결국 1층 선형 변환과 같음을 직접 보여라. (행렬곱의 결합법칙 활용)

2. **미분값 비교**: z = -2, 0, 2에서 Sigmoid, Tanh, ReLU의 출력값과 미분값을 각각 계산하라. 어떤 함수가 z=2에서 가장 큰 기울기를 가지는가?

3. **Dead Neuron 시나리오**: 학습률이 매우 큰 상태에서 ReLU를 사용하면 왜 많은 뉴런이 죽을 수 있는지 설명하라. LeakyReLU가 이 문제를 어떻게 완화하는지도 설명하라.

4. **기울기 소실 계산**: Sigmoid 활성화 함수를 사용하는 20층 네트워크에서, 1층까지 전달되는 기울기의 최대 크기를 추정하라. (힌트: $0.25^{19}$)

5. **GELU vs SiLU**: 두 함수 모두 "부드러운 게이팅" 효과가 있다. GELU는 $z \cdot \Phi(z)$, SiLU는 $z \cdot \sigma(z)$이다. $z = -1, 0, 1$에서 두 함수의 출력을 대략적으로 비교하고, 왜 실무에서 큰 차이가 없는지 논의하라.

6. **VLA 설계 추론**: Llama 2 기반 VLA 모델이 SiLU를 사용하는 이유를 추론해보라. (힌트: Backbone LLM의 선택을 그대로 따르는 것이 일반적이다)

---

## 다음 노트 (Next Note)

활성화 함수를 통해 신경망이 복잡한 함수를 표현할 수 있게 되었다. 하지만 가중치를 **어떻게 업데이트**하는지를 아직 다루지 않았다. 그 핵심이 역전파(Backpropagation)이다.

**다음**: [역전파 (Backpropagation)](./03-backpropagation.md) - 손실에서 출발하여 모든 가중치의 기울기를 효율적으로 계산하는 방법. PyTorch autograd의 원리이기도 하다.
