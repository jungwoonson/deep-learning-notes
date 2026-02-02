# 평가 지표 (Evaluation Metrics)

## 왜 알아야 하는가 (Why This Matters for VLA)

모델을 학습시킨 후 "이 모델이 좋은가?"를 판단하려면 **적절한 평가 지표(metric)** 를 선택해야 한다. 잘못된 지표를 사용하면 실제로는 나쁜 모델을 좋다고 착각할 수 있다.

VLA와의 연결 고리:
- VLA의 Vision 파트 성능을 평가할 때 **precision, recall, mAP** 등을 사용한다
- VLA의 Action 파트는 **회귀 지표(MSE, MAE)** 로 평가한다
- 로봇이 물체를 99% 정확히 인식해도 1%의 실패가 물리적 사고를 일으킬 수 있다 → **어떤 지표가 중요한지**는 응용 분야에 따라 다르다
- VLA 논문을 읽을 때 "success rate", "task completion rate" 등의 지표를 이해하려면 기본 평가 지표를 알아야 한다
- 모델을 비교하고 개선 방향을 결정하는 **모든 의사 결정의 기준**이 평가 지표이다

---

## 핵심 개념 (Core Concepts)

### 1. 분류 평가 지표 (Classification Metrics)

#### 혼동 행렬 (Confusion Matrix)

분류 모델의 모든 예측 결과를 한눈에 보여주는 표이다.

```
                     예측 (Predicted)
                   Positive    Negative
실제   Positive  |   TP     |    FN     |
(Actual)          |          |           |
       Negative  |   FP     |    TN     |

TP (True Positive):  양성을 양성으로 올바르게 예측 (맞춤!)
TN (True Negative):  음성을 음성으로 올바르게 예측 (맞춤!)
FP (False Positive): 음성을 양성으로 잘못 예측 (거짓 경보, Type I Error)
FN (False Negative): 양성을 음성으로 잘못 예측 (놓침, Type II Error)
```

**예시: 암 진단 모델**
```
100명의 환자 중 실제 암 환자 10명

                     예측
                   암       정상
실제   암       |  8 (TP) |  2 (FN)  |    → 암 환자 10명 중 8명 발견
       정상     |  5 (FP) | 85 (TN)  |    → 정상 90명 중 5명을 암으로 오진
```

#### 정확도 (Accuracy)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

위 예시: (8 + 85) / 100 = 93%
```

**정확도의 함정 (Accuracy Paradox)**:
```
클래스 불균형 예시: 1,000명 중 암 환자 5명

"모든 사람을 정상으로 예측하는 모델":
Accuracy = 995/1000 = 99.5%

→ 정확도는 99.5%이지만 암 환자를 단 한 명도 찾지 못함!
→ 불균형 데이터에서 정확도만 보면 안 된다
```

#### 정밀도 (Precision)

"양성으로 예측한 것 중 실제로 양성인 비율" (= 예측의 정확성)

```
Precision = TP / (TP + FP)

위 암 진단 예시: 8 / (8 + 5) = 61.5%

의미: "암이라고 진단한 13명 중 실제 암 환자는 8명 (61.5%)"
→ "모델이 양성이라고 할 때, 얼마나 믿을 수 있는가?"
```

**Precision이 중요한 경우**: 잘못된 양성 판정의 비용이 클 때
- 스팸 필터: 정상 메일을 스팸으로 분류하면 중요한 메일을 놓침
- 추천 시스템: 관련 없는 콘텐츠를 추천하면 사용자 이탈

#### 재현율 (Recall / Sensitivity / True Positive Rate)

"실제 양성 중 모델이 찾아낸 비율" (= 발견 능력)

```
Recall = TP / (TP + FN)

위 암 진단 예시: 8 / (8 + 2) = 80%

의미: "실제 암 환자 10명 중 8명을 발견함 (80%)"
→ "실제 양성을 얼마나 빠짐없이 찾아내는가?"
```

**Recall이 중요한 경우**: 놓치는 것의 비용이 클 때
- 암 진단: 암 환자를 정상으로 판정하면 치료 시기를 놓침
- 결함 검출: 불량품을 출하하면 사고 발생

#### Precision-Recall 트레이드오프

```
분류 임계값(threshold)을 조절하면 Precision과 Recall이 반대로 변한다:

임계값 높임 (예: 0.9 이상만 양성):
  → Precision ↑ (확실한 것만 양성으로 → 정확도 높아짐)
  → Recall ↓ (기준이 엄격해서 많은 양성을 놓침)

임계값 낮춤 (예: 0.3 이상이면 양성):
  → Precision ↓ (기준이 느슨해서 오탐 증가)
  → Recall ↑ (웬만한 양성은 다 잡아냄)

Precision
  |\.
  |  '-.
  |     '-.
  |        '-.
  +----------'→ Recall
```

#### F1 Score

Precision과 Recall의 **조화 평균(harmonic mean)** 이다.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

위 암 진단 예시: 2 * (0.615 * 0.80) / (0.615 + 0.80) = 0.696
```

**왜 산술 평균이 아닌 조화 평균인가?**
```
산술 평균: (Precision + Recall) / 2
  → Precision = 1.0, Recall = 0.0 이면 평균 = 0.5 (너무 관대)

조화 평균: 2 * P * R / (P + R)
  → Precision = 1.0, Recall = 0.0 이면 F1 = 0.0 (적절한 페널티)

→ 조화 평균은 두 값 중 낮은 쪽에 더 많은 가중치를 부여한다
→ 둘 다 높아야 F1도 높아진다
```

### 2. ROC 곡선과 AUC (ROC Curve and AUC)

#### ROC 곡선 (Receiver Operating Characteristic Curve)

분류 임계값을 0에서 1까지 변화시키면서 TPR과 FPR을 그린 곡선이다.

```
TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)
```

```
TPR(Recall)
 1.0 |      .----
     |     /
     |    /
     |   /      ← 좋은 모델 (왼쪽 위에 가까움)
     |  /
 0.5 | /  /     ← 랜덤 모델 (대각선)
     |/  /
     | /
 0.0 +----------
     0.0       1.0
         FPR
```

#### AUC (Area Under the Curve)

ROC 곡선 아래의 면적이다.

```
AUC = 1.0: 완벽한 모델
AUC = 0.5: 랜덤 추측 (동전 던지기)
AUC < 0.5: 랜덤보다 나쁨 (예측을 뒤집으면 나아짐)

좋은 모델:      보통 모델:      나쁜 모델:
AUC ≈ 0.95     AUC ≈ 0.75     AUC ≈ 0.55
```

**AUC의 장점**:
- 임계값에 독립적 → 모델 자체의 분류 능력을 평가
- 클래스 불균형에도 비교적 안정적
- 서로 다른 모델을 공정하게 비교할 수 있음

### 3. 회귀 평가 지표 (Regression Metrics)

#### MSE (Mean Squared Error)
```
MSE = (1/n) * Sigma[(y - y_hat)^2]

- 큰 오차에 큰 페널티
- 단위가 원래 값의 제곱 (해석이 어려움)
```

#### RMSE (Root Mean Squared Error)
```
RMSE = sqrt(MSE)

- MSE에 루트를 씌워 원래 단위로 복원
- "평균적으로 RMSE만큼 틀린다"고 해석 가능
- 예: 집값 예측 RMSE = 5000만원 → "평균 5000만원 정도 차이남"
```

#### MAE (Mean Absolute Error)
```
MAE = (1/n) * Sigma[|y - y_hat|]

- 직관적 해석 가능
- 이상치에 덜 민감
```

#### R^2 (결정 계수, Coefficient of Determination)
```
R^2 = 1 - (Sigma[(y - y_hat)^2] / Sigma[(y - y_mean)^2])

- 모델이 데이터의 변동성을 얼마나 설명하는지 나타냄
- R^2 = 1: 완벽한 예측
- R^2 = 0: 평균값으로만 예측한 것과 같은 수준
- R^2 < 0: 평균값보다 못한 모델 (매우 나쁨)
```

### 4. 다중 분류에서의 지표 (Metrics for Multi-class)

클래스가 3개 이상일 때 Precision, Recall, F1을 계산하는 방법:

#### Macro Average
```
각 클래스별 지표를 계산한 후 단순 평균

Macro F1 = (F1_class1 + F1_class2 + ... + F1_classK) / K

특징: 모든 클래스를 동등하게 취급
     → 소수 클래스의 성능이 잘 반영됨
```

#### Weighted Average
```
각 클래스의 샘플 수에 비례하여 가중 평균

Weighted F1 = Sigma[n_k/N * F1_class_k]

특징: 데이터가 많은 클래스의 성능이 더 크게 반영됨
```

#### Micro Average
```
전체 TP, FP, FN을 합산한 후 지표를 계산

Micro Precision = Sigma[TP_k] / Sigma[TP_k + FP_k]

특징: 다중 분류에서 Micro Average = Accuracy
```

### 5. 평가 지표 선택 가이드 (Which Metric to Use?)

```
상황                              → 추천 지표
───────────────────────────────────────────────
클래스 균형, 일반적 분류            → Accuracy, F1
클래스 불균형                      → F1, AUC, Precision/Recall
놓치면 안 되는 경우 (암 진단)       → Recall 중심
오탐이 비싼 경우 (스팸 필터)        → Precision 중심
모델 비교 (임계값 무관)            → AUC
회귀 (일반)                        → RMSE, R^2
회귀 (이상치 있음)                  → MAE
VLA Action 평가                    → MSE, MAE, Success Rate
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **혼동 행렬 해석**: 다음 혼동 행렬에서 Accuracy, Precision, Recall, F1을 계산하라.
   ```
                  예측: 양성   예측: 음성
   실제: 양성       40          10
   실제: 음성       20          130
   ```

2. **지표 선택**: 자율주행 자동차의 보행자 감지 시스템에서 Precision과 Recall 중 어느 것이 더 중요한가? 이유를 설명하라.

3. **Accuracy의 함정**: 신용카드 사기 탐지에서 전체 거래 중 사기는 0.1%이다. "모든 거래를 정상으로 예측"하는 모델의 Accuracy는? 이 모델이 쓸모 있는가?

4. **F1 계산 연습**: Precision = 0.9, Recall = 0.3일 때와 Precision = 0.6, Recall = 0.6일 때의 F1을 각각 계산하고 비교하라.

5. **VLA 평가 설계**: VLA 로봇이 "물체를 집어서 상자에 넣는" 작업을 수행한다. 이 로봇의 성능을 평가하기 위해 어떤 지표들을 사용해야 할까? (힌트: 물체 인식 성능, 집기 성공률, 위치 정확도 각각에 다른 지표 필요)

6. **ROC/AUC 비교**: 모델 A의 AUC가 0.85이고, 모델 B의 AUC가 0.92이다. 하지만 특정 임계값에서 모델 A의 F1이 더 높을 수 있는가? 왜 그런가?

---
