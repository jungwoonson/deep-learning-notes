# 저장, 로드, GPU 훈련 (Saving, Loading, GPU Training)

## 왜 필요한가 — VLA와의 연결

VLA 모델은 수십억 개의 파라미터를 가지며, 학습에 수백 GPU-시간이 소요된다. 이런 환경에서는 세 가지가 필수다.

첫째, **체크포인트 저장과 로드**. 학습 도중 하드웨어 장애가 발생해도 마지막 체크포인트에서 재개할 수 있어야 한다. 또한 사전 학습된 vision encoder나 language model의 가중치를 로드하여 fine-tune하는 것이 VLA 학습의 기본 패턴이다.

둘째, **GPU 활용**. CPU만으로는 VLA 규모의 모델을 학습하는 것이 사실상 불가능하다. `.to(device)` 한 줄이 전부처럼 보이지만, 실제로는 mixed precision, gradient accumulation 등 GPU 메모리를 효율적으로 사용하는 기법까지 알아야 한다.

셋째, **재현성**. 저장/로드가 올바르게 동작해야 실험 결과를 재현하고, 모델을 배포할 수 있다.

---

## 핵심 개념

### 1. state_dict

`nn.Module`의 모든 학습 가능한 파라미터와 버퍼는 `state_dict()`라는 OrderedDict로 접근할 수 있다.

```python
print(model.state_dict().keys())
# odict_keys(['linear1.weight', 'linear1.bias', 'linear2.weight', ...])
```

`state_dict`는 모델 구조(코드)와 분리된 순수한 가중치 데이터다. 따라서 모델 클래스 정의가 있으면 가중치만 로드하여 복원할 수 있다.

### 2. torch.save와 torch.load

**권장 패턴 — state_dict 저장**:

```python
# 저장
torch.save(model.state_dict(), "model_weights.pt")

# 로드
model = MyModel()  # 같은 구조의 모델 인스턴스 생성
model.load_state_dict(torch.load("model_weights.pt"))
```

**전체 모델 저장** (`torch.save(model, ...)`)은 가능하지만 권장하지 않는다. pickle에 의존하므로, 코드 구조가 바뀌면 로드가 실패한다.

### 3. 체크포인트 (Checkpoint)

학습 재개를 위해서는 모델 가중치뿐 아니라 optimizer 상태, scheduler 상태, epoch 번호 등도 함께 저장해야 한다.

```python
# 저장
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
}
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")

# 로드
checkpoint = torch.load("checkpoint_epoch_5.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

VLA 학습에서는 주기적으로(예: 매 1,000 step) 체크포인트를 저장하고, 디스크 공간 관리를 위해 최근 N개만 유지하는 전략을 사용한다.

### 4. .to(device)와 GPU 활용

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for batch in loader:
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    outputs = model(inputs)
```

**핵심 규칙**: 모델과 데이터가 반드시 같은 device에 있어야 한다. 하나라도 다르면 `RuntimeError`가 발생한다.

`torch.load` 시 device를 지정하지 않으면 저장 당시의 device로 로드된다. GPU에서 저장한 모델을 CPU에서 로드하려면:

```python
state = torch.load("model.pt", map_location="cpu")
```

### 5. Mixed Precision Training (혼합 정밀도 훈련)

기본적으로 PyTorch는 `float32`로 연산한다. **Mixed precision**은 연산의 일부를 `float16` 또는 `bfloat16`으로 수행하여 메모리를 절약하고 속도를 높이는 기법이다.

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

- **`autocast()`**: forward 연산을 자동으로 적절한 저정밀도 타입으로 캐스팅한다.
- **`GradScaler`**: `float16` 사용 시 gradient가 너무 작아지는(underflow) 문제를 loss scaling으로 방지한다. `bfloat16` 사용 시에는 GradScaler가 필수는 아니다.

VLA 모델은 크기가 크므로 mixed precision이 거의 필수다. 메모리 사용량이 절반 가까이 줄어든다.

### 6. Gradient Accumulation (그래디언트 누적)

GPU 메모리에 한 번에 올릴 수 있는 batch_size가 작을 때, 여러 mini-batch의 gradient를 누적한 후 한 번에 업데이트하여 **실질적인 batch size를 키우는** 기법이다.

```python
accumulation_steps = 4

for i, batch in enumerate(loader):
    outputs = model(batch["input"])
    loss = criterion(outputs, batch["target"])
    loss = loss / accumulation_steps  # 누적 횟수로 나누기
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

VLA 학습에서 effective batch size 256이 필요하지만 GPU당 32밖에 못 올리면, `accumulation_steps=8`로 설정하여 동일한 효과를 낸다.

---

## 연습 주제

1. 학습한 모델의 `state_dict()`를 저장하고, 새 모델 인스턴스에 로드한 뒤, 동일한 입력에 대해 같은 출력이 나오는지 확인해 보라.
2. optimizer와 scheduler 상태까지 포함하는 full checkpoint를 저장/로드하고, 학습을 재개해 보라.
3. GPU가 가능한 환경에서 `.to(device)`를 적용하고, 학습 속도 차이를 측정해 보라.
4. `map_location` 파라미터를 사용하여 GPU 체크포인트를 CPU에서 로드해 보라.
5. `torch.cuda.amp.autocast()`를 적용한 학습과 적용하지 않은 학습의 메모리 사용량을 비교해 보라 (`torch.cuda.memory_allocated()` 사용).
6. gradient accumulation을 직접 구현하고, 단일 배치 학습 대비 실질적 batch size가 커졌을 때 loss 곡선이 어떻게 변하는지 비교해 보라.

---
