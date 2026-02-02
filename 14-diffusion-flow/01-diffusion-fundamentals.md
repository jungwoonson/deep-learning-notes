# Diffusion Fundamentals

## VLA 연결고리

Diffusion model은 원래 이미지 생성(Stable Diffusion, DALL-E)으로 유명하지만, 로봇 행동 생성에도 강력하다. **Diffusion Policy**는 로봇의 연속 행동을 diffusion으로 생성하여, action tokenization의 해상도 한계와 MSE regression의 평균 붕괴 문제를 동시에 해결한다. pi-0, Octo 등 최신 VLA가 이 방식을 채택한다.

---

## 핵심 개념

### 1. Generative Models 개관

데이터의 분포를 학습하여 새로운 샘플을 생성하는 모델들이다.

| 모델 | 핵심 아이디어 | 장점 | 단점 |
|------|--------------|------|------|
| GAN | Generator vs Discriminator 경쟁 | 빠른 생성, 고품질 | 학습 불안정, mode collapse |
| VAE | 잠재 공간 학습 + 재구성 | 안정적 학습, 잠재 표현 | 흐릿한 출력 |
| Autoregressive | 한 요소씩 순차 생성 | 정확한 likelihood | 느린 생성 |
| **Diffusion** | 노이즈 제거 과정으로 생성 | 고품질, 안정적, 다양성 | 느린 생성 (개선 중) |

Diffusion model은 2020년 이후 이미지 생성의 주류가 되었고, 이제 로봇 행동 생성으로 확장되고 있다.

### 2. Forward Process (노이즈 추가)

깨끗한 데이터에 점진적으로 노이즈를 추가하여 순수 가우시안 노이즈로 만드는 과정이다.

**과정**:

$$x_0 \text{ (원본)} \to x_1 \text{ (약간 노이즈)} \to x_2 \to \cdots \to x_T \text{ (순수 노이즈)}$$

각 단계에서 가우시안 노이즈를 조금씩 추가한다:

$$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \epsilon$$

여기서 $\epsilon$은 표준 가우시안 노이즈, $\alpha_t$는 noise schedule 파라미터이다.

**중요한 성질**: 임의의 $t$에 대해, $x_0$에서 $x_t$로 한 번에 점프할 수 있다 (reparameterization):

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

이것 덕분에 학습 시 모든 time step을 시뮬레이션할 필요 없이, 랜덤한 $t$를 골라 바로 $x_t$를 만들 수 있다.

### 3. Reverse Process (노이즈 제거)

Forward process를 거꾸로 수행하여 노이즈에서 깨끗한 데이터를 복원하는 과정이다. 이것이 "생성"에 해당한다.

$$x_T \text{ (순수 노이즈)} \to x_{T-1} \to \cdots \to x_1 \to x_0 \text{ (생성된 데이터)}$$

각 단계에서 neural network가 현재 노이즈가 섞인 데이터를 보고, 노이즈를 조금 제거한다. 이 과정을 T번 반복하면 깨끗한 데이터가 생성된다.

**핵심**: Reverse process는 forward process의 역(posterior)이며, 이를 neural network로 근사한다.

### 4. DDPM (Denoising Diffusion Probabilistic Models)

2020년 Ho et al.이 발표한 논문으로, 현대 diffusion model의 기반이다.

**학습 목표**: Network $\epsilon_\theta$가 $x_t$에서 추가된 노이즈 $\epsilon$을 예측하도록 학습한다.

$$\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

즉, "이 노이즈 섞인 데이터에서 노이즈 부분만 찾아내라"가 학습 목표이다.

**학습 과정**:
1. 학습 데이터에서 $x_0$를 가져온다
2. 랜덤한 time step $t$를 고른다
3. 랜덤 노이즈 $\epsilon$을 생성한다
4. $x_t$를 계산한다 (forward process 한 번에 점프)
5. Network가 $x_t$와 $t$를 입력받아 $\epsilon$을 예측한다
6. 예측과 실제 $\epsilon$의 MSE를 최소화한다

**생성 과정**:
1. 순수 노이즈 $x_T$를 샘플링한다
2. $t = T, T-1, \ldots, 1$에 대해 반복:
   - Network로 노이즈를 예측한다
   - 예측된 노이즈를 제거하고 약간의 새 노이즈를 추가한다
3. $x_0$가 생성된 데이터이다

### 5. Noise Schedule

Forward process에서 각 time step마다 얼마나 노이즈를 추가할지 결정하는 스케줄이다.

**Linear schedule**: $\beta_t$가 시간에 따라 선형으로 증가한다. 초기에는 적게, 후반에는 많이 추가.

**Cosine schedule**: Cosine 함수를 사용하여 더 부드러운 노이즈 추가. DDPM 이후 개선된 방법.

**스케줄이 중요한 이유**:
- 너무 빠르게 노이즈를 추가하면: 초반 step에서 정보가 빨리 사라져 복원이 어려움
- 너무 느리게 추가하면: 많은 step이 필요해 비효율적
- 적절한 스케줄이 생성 품질에 큰 영향을 미침

### 6. 학습 목표의 세 가지 동등한 관점

DDPM의 학습 목표는 세 가지 다른 방식으로 해석할 수 있다. 수학적으로 동등하다.

| 예측 대상 | 의미 | 수식 |
|-----------|------|------|
| **epsilon-prediction** | 추가된 노이즈를 예측 | $\epsilon_\theta(x_t, t) \approx \epsilon$ |
| **x_0-prediction** | 원본 데이터를 직접 예측 | $x_{0,\theta}(x_t, t) \approx x_0$ |
| **score prediction** | 데이터 분포의 기울기를 예측 | $s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t)$ |

어떤 관점을 선택하느냐에 따라 구현 방식과 성능이 약간 다르지만, 이론적으로는 같은 것을 학습한다. Score prediction 관점은 다음 노트(Score Matching)에서 다룬다.

---

## 연습 주제 (코드 없이)

1. Forward process에서 T=1000이고 x_0가 이미지일 때, x_500은 어떻게 보일지 상상해 보라. 원본 형태가 어느 정도 남아 있을까?
2. Reverse process에서 network가 노이즈를 완벽하게 예측한다면, 이론적으로 몇 step 만에 x_0를 복원할 수 있는가? (힌트: reparameterization trick)
3. DDPM의 loss가 단순한 MSE인데도 GAN보다 학습이 안정적인 이유를 직관적으로 설명하라.
4. 이미지 생성에서 diffusion이 하는 일(노이즈 -> 이미지)과, 로봇 행동 생성에서 하는 일(노이즈 -> action)의 구조적 유사성을 설명하라.
5. Noise schedule이 너무 공격적(빨리 노이즈 추가)이면 생성 품질에 어떤 영향이 있을지 생각하라.

---
