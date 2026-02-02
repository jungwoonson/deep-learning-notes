# 디버깅 (Debugging)

## 왜 필요한가 — VLA와의 연결

VLA 모델은 vision encoder, language model, action head라는 서로 다른 세 갈래의 파이프라인이 합류하는 구조다. 각 모듈이 개별적으로는 잘 동작해도, 합쳐진 순간 shape 불일치, device 불일치, gradient 소실 등 예상치 못한 문제가 터진다. 게다가 모델 크기가 크므로, 한 번의 학습 실행에 수 시간이 걸리는 경우가 흔하다. 3시간 돌린 뒤에야 NaN loss를 발견하는 상황은 치명적이다.

체계적인 디버깅 습관을 처음부터 들여 놓으면, VLA 규모의 프로젝트에서도 문제를 빠르게 찾고 해결할 수 있다. 이 노트는 PyTorch에서 가장 자주 만나는 오류 유형과 그에 대한 진단 전략을 다룬다.

---

## 핵심 개념

### 1. Shape Mismatch (형태 불일치)

가장 흔한 에러다. 에러 메시지는 보통 이런 형태를 띤다:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 256x64)
```

**원인**: 레이어의 입력 차원과 실제 데이터의 차원이 맞지 않는다.

**진단 전략**:
- forward 메서드 중간중간에 `print(x.shape)`을 넣어 각 레이어 통과 후의 shape을 추적한다.
- 모델 전체를 돌리기 전에, 작은 dummy input으로 forward pass만 먼저 테스트한다.

```python
dummy = torch.randn(1, 3, 224, 224)  # batch=1인 테스트 입력
output = model(dummy)
print(output.shape)
```

VLA에서 특히 주의할 지점: vision encoder 출력의 shape이 language model 입력 projection과 맞는지, action head 출력 차원이 실제 action space와 맞는지 확인해야 한다.

### 2. Device Mismatch (장치 불일치)

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

**원인**: 모델은 GPU에 있는데 데이터가 CPU에 있거나, 중간에 새로 생성한 tensor가 기본값인 CPU에 만들어졌다.

**진단 전략**:
- 에러 발생 직전에 관련 tensor의 `.device`를 출력한다.
- forward 내부에서 새 tensor를 생성할 때는 반드시 기존 tensor와 같은 device에 만든다.

```python
# 잘못된 방법
mask = torch.ones(batch_size, seq_len)  # CPU에 생성됨

# 올바른 방법
mask = torch.ones(batch_size, seq_len, device=x.device)
```

### 3. Gradient 관련 문제

#### gradient가 None

```python
loss.backward()
print(param.grad)  # None
```

**원인**:
- 해당 파라미터가 loss 계산에 관여하지 않았다 (computational graph에 연결되지 않음).
- `requires_grad=False`로 설정되어 있다.
- 중간에 `.detach()`나 `.item()`, `.numpy()` 같은 graph를 끊는 연산이 있었다.

#### gradient가 NaN 또는 Inf

**원인**: 학습률이 너무 높거나, loss function에서 `log(0)` 같은 수치적 불안정이 발생했다.

**진단 전략**:

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")
```

#### Gradient Vanishing / Exploding

깊은 네트워크에서 gradient가 너무 작아지거나(vanishing) 너무 커지는(exploding) 현상.

**진단**: 각 레이어의 gradient 크기(norm)를 모니터링한다.

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
```

**해결**: gradient clipping을 적용한다.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

VLA 모델처럼 깊고 복잡한 구조에서는 gradient clipping이 거의 표준이다.

### 4. Loss가 줄어들지 않는 경우

**체크리스트**:
1. Learning rate가 적절한가? (너무 크면 발산, 너무 작으면 정체)
2. `optimizer.zero_grad()`를 호출했는가?
3. `loss.backward()` 후 `optimizer.step()`을 호출했는가?
4. 데이터 레이블이 올바른가? (데이터 자체가 잘못되면 모델은 학습할 수 없다)
5. 모델이 충분히 표현력이 있는가? (너무 작은 모델은 복잡한 데이터를 학습하지 못한다)
6. Overfit 테스트: 단 1~2개 배치에 대해 loss가 0에 수렴하는지 확인한다. 이것조차 안 되면 코드에 버그가 있다.

### 5. Gradient Checking (수치적 검증)

자동 미분의 결과가 올바른지 유한차분법(finite difference)으로 검증한다.

```python
torch.autograd.gradcheck(
    func, inputs, eps=1e-6, atol=1e-4, rtol=1e-3
)
```

이 방법은 커스텀 autograd Function을 구현했을 때 특히 유용하다. 다만 매우 느리므로 전체 모델이 아닌 개별 함수 단위로 사용한다.

### 6. 메모리 프로파일링

GPU 메모리 부족(`CUDA out of memory`)은 VLA 규모에서 일상적으로 마주하는 문제다.

**진단 도구**:

```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
```

**흔한 원인과 해결**:

| 원인 | 해결 |
|------|------|
| 배치 크기가 너무 큼 | batch_size 줄이기, gradient accumulation 적용 |
| 불필요한 tensor가 GPU에 남아 있음 | `del tensor`, `torch.cuda.empty_cache()` |
| 검증 시 gradient graph가 유지됨 | `torch.no_grad()` 블록 사용 |
| activation이 모두 메모리에 저장됨 | gradient checkpointing (`torch.utils.checkpoint`) 적용 |
| `float32`로 학습 중 | mixed precision 적용 |

### 7. 재현 가능한 실험을 위한 시드 고정

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

디버깅 시 실행마다 결과가 달라지면 원인 파악이 매우 어렵다. 시드를 고정하면 동일한 조건에서 동일한 결과를 얻을 수 있다. 다만 `num_workers > 0`이거나 CUDA 비결정적 연산이 포함되면 완벽한 재현이 보장되지 않을 수 있다.

---

## 연습 주제

1. 의도적으로 `nn.Linear`의 입력 차원을 잘못 설정하고, 에러 메시지를 읽고 수정해 보라.
2. 모델은 GPU에, 데이터는 CPU에 두고 forward pass를 실행하여 device mismatch 에러를 확인한 뒤 해결해 보라.
3. 간단한 모델로 overfit 테스트를 수행해 보라: 단일 배치에 대해 loss가 0에 수렴하는지 확인하라.
4. gradient norm을 매 step 출력하면서 학습하고, gradient가 어느 레이어에서 가장 큰지/작은지 분석해 보라.
5. `torch.cuda.memory_allocated()`를 학습 루프 전후에 출력하며, 어떤 연산이 메모리를 가장 많이 사용하는지 파악해 보라.
6. `torch.autograd.gradcheck`을 사용하여 직접 만든 간단한 함수의 gradient가 올바른지 검증해 보라.

---
