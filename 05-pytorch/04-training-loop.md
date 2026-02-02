# 훈련 루프 (Training Loop)

## 왜 필요한가 — VLA와의 연결

VLA 모델 학습은 한 줄의 `.fit()` 호출로 끝나지 않는다. Vision encoder는 freeze한 채 language model만 fine-tune하거나, action head에만 다른 learning rate를 적용하거나, 특정 단계에서 학습 전략을 바꾸는 등 세밀한 제어가 필요하다. PyTorch는 의도적으로 고수준 학습 API를 제공하지 않으며, 대신 **명시적인 훈련 루프**를 사용자가 직접 작성하도록 설계되어 있다.

이 "직접 작성하는 루프"는 처음에는 번거롭게 느껴질 수 있지만, VLA처럼 복잡한 모델을 다룰 때는 오히려 유연함이 가장 큰 장점이 된다. 정규(canonical) 훈련 루프의 구조를 완전히 이해하면, 어떤 모델이든 학습시킬 수 있다.

---

## 핵심 개념

### 1. 정규 훈련 루프 (Canonical Training Loop)

PyTorch 훈련의 핵심은 다섯 단계의 반복이다.

```
forward → loss 계산 → zero_grad → backward → optimizer step
```

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # 1. Forward: 입력을 모델에 통과
        outputs = model(batch["input"])

        # 2. Loss 계산
        loss = criterion(outputs, batch["target"])

        # 3. Gradient 초기화 (누적 방지)
        optimizer.zero_grad()

        # 4. Backward: gradient 계산
        loss.backward()

        # 5. Step: 파라미터 업데이트
        optimizer.step()
```

**순서가 중요하다**: `zero_grad()`는 반드시 `backward()` 전에 호출해야 한다. 그렇지 않으면 이전 배치의 gradient가 누적되어 학습이 올바르게 진행되지 않는다.

### 2. Loss Function (손실 함수)

PyTorch는 `torch.nn` 모듈에 다양한 loss를 제공한다.

| Loss | 용도 | VLA 관련 |
|------|------|----------|
| `nn.CrossEntropyLoss` | 분류 | 다음 토큰 예측 (language model) |
| `nn.MSELoss` | 회귀 | 연속 행동 값 예측 (action head) |
| `nn.L1Loss` | 회귀 (절대 오차) | 행동 예측의 대안 |
| `nn.BCEWithLogitsLoss` | 이진 분류 | 그리퍼 열림/닫힘 예측 |

VLA 모델은 보통 여러 loss를 결합한다. 예를 들어 language modeling loss와 action prediction loss를 가중합으로 합산한다.

### 3. Optimizer (옵티마이저)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

- **SGD**: 가장 기본. momentum과 함께 사용.
- **Adam**: adaptive learning rate. 대부분의 딥러닝 실험에서 기본 선택.
- **AdamW**: Adam에 decoupled weight decay를 적용. Transformer 계열 모델의 표준.

VLA 학습에서 서브 모듈별로 다른 learning rate를 적용하려면 parameter group을 분리한다.

```python
optimizer = AdamW([
    {"params": model.vision_encoder.parameters(), "lr": 1e-5},
    {"params": model.language_model.parameters(), "lr": 1e-4},
    {"params": model.action_head.parameters(), "lr": 3e-4},
])
```

### 4. Epoch vs Iteration

- **Epoch**: 전체 학습 데이터를 한 번 순회하는 것.
- **Iteration (Step)**: 배치 하나를 처리하는 것.

전체 데이터가 10,000개이고 batch_size가 32이면, 1 epoch = 약 312 iteration이다. VLA처럼 대규모 데이터셋에서는 1 epoch도 매우 오래 걸리므로, iteration 단위로 학습을 관리하는 경우가 많다.

### 5. 검증 루프 (Validation Loop)

```python
model.eval()
val_loss = 0.0
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch["input"])
        loss = criterion(outputs, batch["target"])
        val_loss += loss.item()
```

**핵심 차이점**:
- `model.eval()` — Dropout, BatchNorm 등의 동작이 추론 모드로 전환된다.
- `torch.no_grad()` — gradient 계산을 비활성화하여 메모리를 절약한다.
- `optimizer.step()` 없음 — 검증은 파라미터를 업데이트하지 않는다.

검증 루프를 빼먹거나, `model.eval()`을 호출하지 않으면 과적합 여부를 정확히 판단할 수 없다.

### 6. Learning Rate Scheduler

학습률을 고정하지 않고 학습 진행에 따라 조절하는 전략이다.

- **StepLR**: 일정 epoch마다 learning rate를 감소.
- **CosineAnnealingLR**: 코사인 곡선을 따라 서서히 감소.
- **Warmup + Cosine Decay**: 초기 몇 스텝 동안 learning rate를 점진적으로 올린 후 감소. Transformer/VLA 학습의 표준.

```python
scheduler.step()  # 보통 epoch 또는 step 단위로 호출
```

### 7. 로깅 (Logging)

학습 과정을 추적하지 않으면 문제 진단이 불가능하다. 최소한 기록해야 할 것:

- 매 N step마다: train loss
- 매 epoch(또는 N step)마다: validation loss, learning rate
- 선택적으로: gradient norm, GPU 메모리 사용량

도구로는 `print`, TensorBoard, Weights & Biases (wandb) 등이 있다.

---

## 연습 주제

1. 간단한 선형 회귀 문제(y = 2x + 1)에 대해 정규 훈련 루프를 처음부터 작성해 보라.
2. `zero_grad()`를 생략했을 때 loss가 어떻게 변화하는지 관찰해 보라.
3. 학습 데이터와 검증 데이터를 분리하고, 매 epoch마다 양쪽 loss를 기록해 보라.
4. `model.eval()`을 호출하지 않았을 때 Dropout이 포함된 모델의 검증 결과가 어떻게 달라지는지 확인해 보라.
5. AdamW에서 learning rate를 10배 키우거나 줄여서 학습 안정성의 차이를 관찰해 보라.
6. CosineAnnealingLR scheduler를 적용하고, 매 step의 learning rate를 출력해 보라.

---

## 다음 노트

[저장, 로드, GPU 훈련](./05-saving-loading-gpu.md) — 학습된 모델을 저장하고 다시 불러오는 방법, 그리고 GPU를 활용한 효율적인 학습 기법을 다룬다.
