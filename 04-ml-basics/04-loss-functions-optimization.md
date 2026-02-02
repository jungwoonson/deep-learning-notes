# 손실 함수와 최적화 (Loss Functions and Optimization)

## 왜 알아야 하는가 (Why This Matters for VLA)

손실 함수(Loss Function)는 모델에게 "얼마나 틀렸는지"를 알려주는 **성적표**이고, 최적화(Optimization)는 그 성적을 올리기 위한 **공부 전략**이다. 이 두 가지가 딥러닝 학습의 핵심 엔진이다.

VLA와의 연결 고리:
- VLA는 **여러 종류의 손실 함수를 동시에** 사용한다
  - Vision: 이미지 분류에 Cross-Entropy, 물체 위치 예측에 MSE 또는 Smooth L1
  - Language: 다음 토큰 예측에 Cross-Entropy
  - Action: 로봇 동작 예측에 MSE 또는 Huber Loss
- VLA처럼 큰 모델은 **mini-batch SGD + 학습률 스케줄링**을 반드시 사용한다
- 최적화 전략이 잘못되면 수억 원의 GPU 비용을 낭비하고도 모델이 학습되지 않을 수 있다

---

## 핵심 개념 (Core Concepts)

### 1. 손실 함수 총정리 (Loss Functions Overview)

손실 함수는 **문제 유형에 따라** 선택한다:

#### 회귀용 손실 함수 (Regression Losses)

**MSE (Mean Squared Error)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

특징:
- 큰 오차에 큰 페널티 (제곱이므로)
- 이상치(outlier)에 민감
- gradient가 오차에 비례하여 크기 때문에 큰 오차를 빠르게 줄임
- 가장 널리 사용되는 회귀 손실 함수

**MAE (Mean Absolute Error)**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

특징:
- 모든 오차에 동일한 가중치
- 이상치에 덜 민감 (robust)
- 0에서 미분 불가능 (실제로는 subgradient 사용)
- gradient가 항상 일정 크기 → 수렴이 느릴 수 있음

**Huber Loss (Smooth L1 Loss)**

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

특징:
- MSE와 MAE의 장점을 결합
- 작은 오차에는 MSE처럼 작동 (매끄러운 gradient)
- 큰 오차에는 MAE처럼 작동 (이상치에 강건)
- 물체 검출(Object Detection)에서 많이 사용
- VLA의 Action 예측에서 자주 선택됨

**세 손실 함수 비교**:
```
손실값
  |      /  MSE (급격히 증가)
  |     /
  |    / / Huber (중간)
  |   / //
  |  / // / MAE (선형 증가)
  | / ///
  |////
  +------------- 오차 크기
```

#### 분류용 손실 함수 (Classification Losses)

**Binary Cross-Entropy (BCE)**

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

용도: 이진 분류 (스팸/정상, 양성/음성), 출력층: sigmoid

**Categorical Cross-Entropy (CCE)**

$$\text{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

용도: 다중 분류 (숫자 인식, 물체 분류), 출력층: softmax

실무 팁: PyTorch에서는 `CrossEntropyLoss`가 softmax + CCE를 합쳐놓은 것 (직접 softmax 적용 불필요)

#### 손실 함수 선택 가이드

| 문제 유형 | 손실 함수 | 출력 활성화 |
|-----------|-----------|------------|
| 이진 분류 | BCE | sigmoid |
| 다중 분류 (하나만 선택) | CCE | softmax |
| 다중 레이블 (여러 개 선택) | BCE (각 클래스별) | sigmoid (각 클래스별) |
| 회귀 (일반) | MSE | 없음 (linear) |
| 회귀 (이상치 있음) | Huber / MAE | 없음 (linear) |

### 2. 경사 하강법의 변형들 (Variants of Gradient Descent)

#### Batch Gradient Descent (전체 배치)

매 업데이트마다 전체 훈련 데이터를 사용:

$$w \leftarrow w - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_w L_i$$

- 장점: 안정적인 수렴, 정확한 gradient
- 단점: 데이터가 크면 매우 느림, 메모리 부족

#### Stochastic Gradient Descent (확률적, SGD)

매 업데이트마다 데이터 1개만 사용:

$$w \leftarrow w - \eta \cdot \nabla_w L_i \quad \text{(랜덤으로 선택한 데이터 } i \text{)}$$

- 장점: 매우 빠른 업데이트, 지역 최솟값 탈출 가능
- 단점: 노이즈가 커서 수렴 경로가 불안정

#### Mini-batch Gradient Descent (미니 배치)

매 업데이트마다 일부(batch_size개) 데이터를 사용:

$$w \leftarrow w - \eta \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla_w L_i$$

- 장점: Batch와 SGD의 장점을 결합, GPU 병렬 처리에 효율적, 적당한 노이즈로 지역 최솟값 탈출 가능
- 단점: batch_size를 적절히 설정해야 함
- → 실무에서 거의 항상 Mini-batch를 사용 (일반적으로 "SGD"라 부르면 이것을 의미)

**비교 시각화**:
```
        Batch GD:          Mini-batch GD:      SGD (단일):
수렴 경로:
        \                  \                    \  /
         \                  \ /                 \/  /\
          \                  \/                     \/\
           \.                 \.                      \.
    (매끄러움)          (약간 흔들림)          (매우 불안정)
```

**일반적인 batch size**: 32, 64, 128, 256
- 작은 batch: 노이즈 많지만 일반화 성능이 좋은 경향
- 큰 batch: 학습이 안정적이지만 더 많은 GPU 메모리 필요

### 3. 에폭과 반복 (Epoch and Iteration)

```
데이터 1,000개, batch_size = 100 일 때:

1 iteration = 100개 데이터로 1번 업데이트
1 epoch = 전체 1,000개 데이터를 한 바퀴 = 10 iterations

전체 학습 = 여러 epoch 반복 (예: 50 epochs = 500 iterations)
```

### 4. 학습률 스케줄 (Learning Rate Schedules)

학습 내내 같은 학습률을 사용하는 것은 비효율적이다.

**비유**: 처음에는 큰 보폭으로 대략적인 위치를 찾고, 나중에는 작은 보폭으로 정밀하게 조정한다.

#### Step Decay

일정 에폭마다 학습률을 일정 비율로 감소:

$$\eta = \eta_0 \times \gamma^{\lfloor \text{epoch} / s \rfloor}$$

예: 매 30 에폭마다 학습률을 절반으로 → $\eta$: $0.01 \to 0.005 \to 0.0025 \to \cdots$

```
lr
 |____
 |    |____
 |         |____
 +------------------ epoch
```

#### Cosine Annealing

코사인 함수를 따라 학습률을 점진적으로 감소:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi \cdot t}{T}\right)\right)$$

```
lr
 |\.
 |  '-.
 |     '-.
 |        '._
 +------------- epoch
```

→ 현대 딥러닝에서 매우 인기 있는 스케줄

#### Warmup + Decay

처음에 학습률을 0에서 서서히 올린 후(warmup), 점차 감소:

```
lr
 | /\
 |/  '-.
 |      '-._
 +------------- epoch
 warmup  decay
```

→ Transformer, VLA 등 대규모 모델에서 거의 필수. 갑자기 큰 학습률로 시작하면 학습 초기에 불안정해지기 때문.

### 5. 학습률이 너무 크거나 작을 때 (Learning Rate Diagnosis)

```
손실(Loss) 그래프로 진단:

이상적:          lr 너무 높음:       lr 너무 낮음:
loss              loss               loss
 |\.               | /\  /\          |\
 |  '-.            |/  \/  \         | \
 |     '-.         |        \        |  \
 |        '-._     |         ...     |   \...............
 +--------epoch    +--------epoch    +--------epoch
(부드럽게 감소)   (진동 또는 발산)    (매우 느린 감소)
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **손실 함수 선택**: 다음 각 상황에서 어떤 손실 함수를 사용해야 하는가?
   - 아파트 가격 예측 (가끔 비정상적으로 비싼 매물이 있음)
   - 이미지에서 고양이/개 분류
   - 영화 리뷰에서 감정 분석 (긍정/부정/중립)
   - 로봇 팔의 목표 위치$(x, y, z)$ 예측

2. **MSE vs MAE 비교**: 실제값이 $[10, 10, 10, 100]$이고 예측값이 $[11, 9, 10, 50]$일 때, MSE와 MAE를 각각 계산하라. 이상치(100)가 각 손실에 얼마나 영향을 미치는가?

3. **Mini-batch 계산**: 훈련 데이터 50,000개, batch_size 128일 때, 1 epoch은 몇 iteration인가? 100 epoch은 총 몇 번 파라미터를 업데이트하는가?

4. **학습률 스케줄 설계**: 총 100 epoch을 학습한다고 할 때, warmup 10 epoch + cosine decay를 적용하면 학습률이 어떻게 변하는지 대략적인 그래프를 그려보라.

5. **VLA 학습 전략**: VLA 모델이 Vision(분류)과 Action(회귀)을 동시에 학습해야 한다면, 각각 다른 손실 함수를 사용하게 된다. 이 두 손실을 어떻게 합칠 수 있을까? (힌트: weighted sum)

---

## 다음 노트 (Next Note)

손실 함수와 최적화 방법을 배웠다. 하지만 학습을 잘 해도 **새로운 데이터에서 성능이 떨어지는** 문제가 있다.

**다음**: [과적합과 정규화 (Overfitting and Regularization)](./05-overfitting-regularization.md) - 모델이 훈련 데이터를 "외워버리는" 문제와 이를 방지하는 기법들을 배운다.
