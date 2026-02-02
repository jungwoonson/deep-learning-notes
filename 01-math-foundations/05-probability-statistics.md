# 확률과 통계 (Probability and Statistics)

## 왜 필요한가?
신경망의 출력은 확률 분포로 해석된다. 분류 모델은 각 클래스의 확률을 출력하고, 생성 모델은 다음 토큰의 확률 분포를 예측한다. 손실 함수(cross-entropy)는 확률 분포 사이의 차이를 측정한다. VLA 모델에서도 액션 토큰의 확률 분포를 예측한다.

---

## 1. 확률 기초

### 확률 (Probability)
어떤 사건이 일어날 가능성. 0(불가능)에서 1(확실) 사이의 값.

$$P(\text{앞면}) = 0.5, \quad P(3) = \frac{1}{6} \approx 0.167$$

### 조건부 확률 (Conditional Probability)
B가 일어났을 때 A가 일어날 확률.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

딥러닝에서: 자기회귀 언어 모델은 이전 토큰들이 주어졌을 때 다음 토큰의 조건부 확률을 예측한다.

$$P(\text{다음 토큰} \mid \text{이전 토큰들})$$

## 2. 확률 분포 (Probability Distribution)

### 이산 분포 (Discrete)
가능한 값이 정해져 있는 경우. 모든 확률의 합 = 1.

$$P(1) = P(2) = P(3) = P(4) = P(5) = P(6) = \frac{1}{6}$$

분류 모델의 출력:

$$P(\text{고양이}) = 0.7, \quad P(\text{개}) = 0.2, \quad P(\text{새}) = 0.1 \quad (\text{합} = 1.0)$$

### 연속 분포 (Continuous)
연속적인 값의 범위. 확률 밀도 함수(PDF)로 표현.

### 정규 분포 (Normal/Gaussian Distribution)
가장 중요한 연속 분포. 평균($\mu$)과 표준편차($\sigma$)로 정의.

$$\mathcal{N}(\mu, \sigma^2)$$

표준 정규 분포: $\mathcal{N}(0, 1)$ — 평균 0, 표준편차 1

딥러닝에서의 사용:
- 가중치 초기화: 정규 분포에서 랜덤 샘플링
- Diffusion 모델: 가우시안 노이즈를 점진적으로 추가/제거
- VAE: 잠재 공간을 정규 분포로 가정

## 3. 기대값과 분산

### 기대값 (Expected Value, $E[X]$)
확률 변수의 "평균적인 값". 각 값 × 확률의 합.

$$E[X] = 1 \times \frac{1}{6} + 2 \times \frac{1}{6} + \cdots + 6 \times \frac{1}{6} = 3.5$$

### 분산 (Variance, $\text{Var}[X]$)
값이 평균으로부터 얼마나 퍼져있는지.

$$\text{Var}[X] = E[(X - E[X])^2]$$

### 표준편차 (Standard Deviation)
분산의 제곱근.

$$\sigma = \sqrt{\text{Var}[X]}$$

딥러닝에서: Batch Normalization은 각 층의 출력을 평균 0, 분산 1로 정규화한다.

## 4. 소프트맥스 (Softmax)
임의의 실수 벡터를 **확률 분포**로 변환하는 함수. 모든 값이 양수, 합이 1.

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

예: $z = [2.0, 1.0, 0.1]$

```
e^z = [7.39, 2.72, 1.11]
합 = 11.22
softmax = [0.659, 0.242, 0.099]   (합 = 1.0)
```

딥러닝에서:
- 분류 모델의 마지막 층: 로짓(logit) → softmax → 확률
- Transformer의 어텐션: $\text{softmax}\!\left(\dfrac{QK^T}{\sqrt{d}}\right)$ — 어텐션 가중치를 확률로 변환
- VLA: 액션 토큰의 확률 분포 예측에 사용

### Softmax의 온도 (Temperature)

$$\text{softmax}\!\left(\frac{z_i}{T}\right)$$

- $T > 1$: 분포가 균일해짐 (탐색적)
- $T < 1$: 분포가 뾰족해짐 (확신적)
- $T \to 0$: argmax와 같아짐 (가장 높은 확률만 선택)

## 5. 최대 우도 추정 (Maximum Likelihood Estimation, MLE)
데이터를 가장 잘 설명하는 파라미터를 찾는 방법.

데이터 $D = \{x_1, x_2, \ldots, x_n\}$이 주어졌을 때, 파라미터 $\theta$를 찾아 $P(D|\theta)$를 최대화.

실제로는 로그를 취해서:

$$\theta^* = \arg\max_\theta \sum_i \log P(x_i | \theta)$$

**핵심 연결**: 신경망 학습에서 **크로스엔트로피 손실 최소화** = **로그 우도 최대화**와 동일하다. 즉, 딥러닝 학습은 MLE를 수행하는 것이다.

## 6. 베이즈 정리 (Bayes' Theorem) — 개념만

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

$$\text{사후확률} = \frac{\text{우도} \times \text{사전확률}}{\text{증거}}$$

딥러닝에서 직접 사용하는 경우는 적지만, 확률적 추론의 기반이 된다.

---

## 핵심 정리
1. **확률 분포**: 모델의 출력은 확률 분포. 분류 = 이산, 생성 = 조건부 확률
2. **정규 분포**: 가중치 초기화, 노이즈, 정규화에 등장
3. **소프트맥스**: 로짓 → 확률 변환. 어텐션과 분류의 핵심
4. **MLE**: 크로스엔트로피 최소화 = 우도 최대화. 학습의 수학적 의미

---

## 실습 과제
1. $[2.0, 1.0, 0.5]$에 softmax를 손으로 계산해보기
2. 온도 $T=0.5, 1.0, 2.0$에서 softmax 결과가 어떻게 달라지는지 비교해보기
3. 동전을 10번 던져 7번 앞면이 나왔을 때, MLE로 앞면 확률을 추정해보기

---

## 다음 노트
→ [정보이론](./06-information-theory.md): 엔트로피, 크로스엔트로피 — 손실 함수의 수학적 기반
