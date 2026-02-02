# Matplotlib 시각화 (Matplotlib Visualization)

## 왜 필요한가: VLA로 가는 길

딥러닝에서 시각화는 선택이 아니라 **디버깅과 이해의 핵심 도구**다. VLA 모델을 다루면서 반드시 시각화해야 하는 것들:

- **Vision**: 입력 이미지가 전처리 후에도 올바른지 확인 (`imshow`)
- **Language**: attention weight이 어떤 토큰에 집중하는지 확인 (heatmap)
- **Action**: 로봇 궤적이 합리적인지 확인 (line/scatter plot)
- **학습 과정**: loss가 줄어드는지, 과적합이 발생하는지 확인 (training curves)

숫자 배열만 보면서 모델을 개발하는 것은 눈을 감고 운전하는 것과 같다. 시각화는 데이터와 모델의 상태를 **직관적으로 이해**하게 해준다.

---

## 핵심 개념

### 1. Matplotlib 기본 구조

Matplotlib은 **Figure**와 **Axes**의 계층 구조로 이루어진다.

- **Figure**: 전체 그림 (캔버스)
- **Axes**: Figure 안의 개별 차트 영역
- 하나의 Figure에 여러 Axes를 배치할 수 있다 (subplots)

```
fig, ax = plt.subplots()           # Figure 1개, Axes 1개
fig, axes = plt.subplots(2, 3)     # Figure 1개, Axes 6개 (2행 3열)
```

`plt.plot()` 같은 간편 함수도 있지만, **`fig, ax` 패턴에 익숙해지는 것이 좋다**. 논문 수준의 그래프를 만들려면 Axes 객체를 직접 다루어야 한다.

### 2. 기본 플롯 (Basic Plots)

#### Line Plot (선 그래프)

연속적인 값의 변화를 보여준다.

```
ax.plot(x, y, label='train loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
```

**VLA 연결**: 학습 중 매 epoch마다 기록한 loss 값을 line plot으로 그려서 학습이 잘 진행되는지 확인한다.

#### Scatter Plot (산점도)

개별 데이터 포인트의 분포를 보여준다.

```
ax.scatter(x, y, c=labels, cmap='viridis', alpha=0.5)
```

**VLA 연결**: 임베딩 벡터를 2D로 축소(t-SNE, PCA)한 뒤 scatter plot으로 클러스터를 확인한다. 비슷한 의미의 토큰이 가까이 모여 있는지 등을 시각적으로 판단할 수 있다.

#### Histogram (히스토그램)

값의 분포를 보여준다.

```
ax.hist(data, bins=50, alpha=0.7)
```

**VLA 연결**: 가중치(weight)의 분포를 확인해 학습이 정상인지 판단한다. 값이 0 근처에 몰려 있으면 vanishing gradient, 극단적으로 퍼져 있으면 exploding gradient를 의심할 수 있다.

### 3. Subplots (다중 차트)

여러 그래프를 한 Figure에 배치한다.

```
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(epochs, train_loss)
axes[0].set_title('Train Loss')

axes[1].plot(epochs, val_loss)
axes[1].set_title('Validation Loss')

axes[2].plot(epochs, accuracy)
axes[2].set_title('Accuracy')

fig.tight_layout()
```

**VLA 연결**: 학습 과정을 모니터링할 때 loss, accuracy, learning rate 등 여러 지표를 한 눈에 비교해야 한다. subplots 없이는 불편하다.

### 4. imshow로 이미지 시각화

NumPy 배열을 이미지로 보여준다. 딥러닝에서 가장 많이 쓰는 시각화 중 하나다.

```
ax.imshow(image_array)                          # RGB 이미지
ax.imshow(gray_image, cmap='gray')              # 흑백 이미지
ax.imshow(feature_map, cmap='viridis')          # feature map
ax.axis('off')                                  # 축 숨기기
```

주의할 점:
- `imshow`는 `(H, W, 3)` (RGB) 또는 `(H, W)` (흑백) 형태를 기대한다
- 값의 범위: `uint8`이면 0~255, `float`이면 0.0~1.0
- PyTorch 텐서 `(C, H, W)`를 `imshow`하려면 `(H, W, C)`로 transpose 필요

**VLA 연결**: 로봇 카메라 입력 이미지를 시각화해서 전처리(resize, crop, 정규화)가 올바른지 확인한다. 전처리가 잘못되면 모델이 아무리 좋아도 제대로 동작하지 않는다.

### 5. Training Curves (학습 곡선)

모델 학습의 상태를 진단하는 가장 기본적인 도구다.

```
ax.plot(epochs, train_loss, label='Train')
ax.plot(epochs, val_loss, label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('Training Curves')
```

학습 곡선에서 읽어야 할 패턴들:

| 패턴 | 의미 | 대응 |
|------|------|------|
| train과 val 모두 감소 | 정상 학습 중 | 계속 학습 |
| train 감소, val 증가 | 과적합 (overfitting) | regularization, early stopping |
| train과 val 모두 높은 수준 | 과소적합 (underfitting) | 모델 크기 증가, 학습률 조정 |
| loss가 진동 | 학습률이 너무 높음 | 학습률 감소 |
| loss가 갑자기 폭등 | 학습 불안정 | gradient clipping, 학습률 감소 |

### 6. Heatmap으로 Attention 시각화

2차원 배열의 값을 색상으로 표현한다.

```
im = ax.imshow(attention_weights, cmap='hot', aspect='auto')
ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45)
ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens)
fig.colorbar(im, ax=ax)
```

**VLA 연결**: Transformer 기반 VLA 모델에서 attention weight을 heatmap으로 시각화하면:
- 어떤 이미지 패치에 주목하는지 확인 (Vision)
- 어떤 단어 간에 관계가 강한지 확인 (Language)
- 언어 지시와 이미지 영역의 cross-attention 확인 (Vision-Language)

이런 시각화가 모델의 동작을 **해석(interpretability)** 하는 핵심 도구다.

### 7. 유용한 팁 모음

```
# 그래프 크기 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 로그 스케일 (loss가 급격히 줄어들 때 유용)
ax.set_yscale('log')

# 격자선
ax.grid(True, alpha=0.3)

# 여러 이미지를 격자로 보기
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(f'Label: {labels[i]}')
    ax.axis('off')
```

---

## 연습 주제

1. **Line Plot 연습**: 임의의 `sin`, `cos` 함수를 한 그래프에 그리고, 범례(legend), 제목, 축 이름을 추가해 보자.

2. **가짜 학습 곡선**: 임의로 train loss(점점 감소)와 val loss(감소 후 증가)를 만들어서 과적합 패턴의 training curve를 그려 보자.

3. **이미지 시각화**: `np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)`로 랜덤 이미지를 만들고 `imshow`로 표시해 보자. 흑백(`cmap='gray'`)도 시도해 보자.

4. **Subplot 격자**: 2x3 subplot 격자를 만들어 6개의 서로 다른 랜덤 이미지를 표시하고, 각각에 제목을 붙여 보자.

5. **Heatmap 연습**: `(10, 10)` 크기의 랜덤 배열을 heatmap으로 표시하고, colorbar를 추가해 보자. `cmap`을 `'hot'`, `'viridis'`, `'coolwarm'` 등으로 바꿔가며 차이를 확인해 보자.

6. **히스토그램으로 분포 비교**: `np.random.randn`으로 평균이 다른 두 분포를 만들고, 같은 axes에 겹쳐서 히스토그램을 그려 보자 (`alpha`를 활용).

---

## 다음 노트

다음은 [Pandas 기초 (Pandas Basics)](./04-pandas-basics.md)를 학습한다. 실제 데이터셋은 CSV, JSON 등 표 형태인 경우가 많다. Pandas로 이런 데이터를 불러오고 탐색하고 전처리하는 방법을 배운다.
