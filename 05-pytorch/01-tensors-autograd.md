# 텐서와 Autograd (Tensors and Autograd)

## 왜 필요한가 — VLA와의 연결

VLA(Vision-Language-Action) 모델은 이미지, 텍스트, 로봇 행동이라는 세 가지 서로 다른 형태의 데이터를 하나의 파이프라인에서 처리한다. 이 모든 데이터는 PyTorch 안에서 **tensor**라는 단일 자료구조로 표현된다. 이미지 픽셀은 `(B, C, H, W)` 형태의 4D tensor, 토큰화된 텍스트는 `(B, T)` 형태의 2D integer tensor, 로봇 관절 명령은 `(B, action_dim)` 형태의 1D~2D tensor가 된다. 따라서 tensor 생성, 변환, 연산을 자유롭게 다루는 것이 VLA 구현의 첫 번째 조건이다.

학습이 이루어지려면 모델이 "얼마나 틀렸는지"를 역방향으로 전파하여 가중치를 갱신해야 한다. PyTorch의 **Autograd** 엔진이 이 역할을 맡는다. VLA처럼 수십억 개의 파라미터가 얽힌 복합 모델에서도 Autograd는 computational graph를 자동으로 구성하고, `.backward()` 한 번으로 모든 gradient를 계산해 준다.

---

## 핵심 개념

### 1. Tensor 생성과 기본 속성

Tensor는 NumPy의 ndarray와 유사하지만, GPU 연산과 자동 미분을 지원한다는 점이 다르다.

- **생성 함수**: `torch.zeros`, `torch.ones`, `torch.rand`, `torch.randn`, `torch.tensor`, `torch.from_numpy`
- **핵심 속성**:
  - `.shape` (또는 `.size()`) — 차원 크기
  - `.dtype` — 데이터 타입 (`torch.float32`, `torch.int64`, `torch.bfloat16` 등)
  - `.device` — 텐서가 위치한 장치 (`cpu` 또는 `cuda:0`)

```python
x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
```

**dtype 선택이 중요한 이유**: VLA 모델 학습에서는 메모리 절약을 위해 `bfloat16`이나 `float16` 혼합 정밀도(mixed precision)를 자주 사용한다. dtype을 잘못 맞추면 연산 오류나 정밀도 손실이 발생한다.

### 2. Tensor 연산과 Shape 변환

- **원소별 연산**: `+`, `-`, `*`, `/`, `torch.exp`, `torch.log`
- **행렬 연산**: `@` (matmul), `torch.bmm` (batched matmul)
- **Shape 변환**: `.view()`, `.reshape()`, `.permute()`, `.transpose()`, `.unsqueeze()`, `.squeeze()`
- **결합/분할**: `torch.cat`, `torch.stack`, `torch.split`, `torch.chunk`
- **Broadcasting**: 차원이 다른 tensor끼리 연산할 때 자동으로 차원을 맞춰 주는 규칙

Shape 변환은 VLA에서 매우 빈번하다. 예를 들어 vision encoder 출력 `(B, num_patches, dim)`을 language model 입력 형태에 맞추려면 `.view()`나 projection layer를 거쳐야 한다.

### 3. Device 이동

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

모델과 데이터가 **같은 device**에 있어야 연산이 가능하다. device mismatch는 가장 흔한 런타임 에러 중 하나다.

### 4. requires_grad와 Computational Graph

`requires_grad=True`로 설정된 tensor에 연산을 수행하면, PyTorch는 내부적으로 **computational graph**(연산 그래프)를 구성한다. 이 그래프는 어떤 연산이 어떤 순서로 적용되었는지를 기록한다.

```python
w = torch.randn(3, 1, requires_grad=True)
y = (x @ w).sum()
```

이때 `y`는 `.grad_fn` 속성을 가지며, 이것이 역전파의 시작점이 된다.

### 5. backward()와 grad

```python
y.backward()   # 역전파 수행
print(w.grad)  # dy/dw 값이 저장됨
```

- `.backward()`는 scalar 값에 대해서만 직접 호출할 수 있다 (non-scalar라면 `gradient` 인자를 전달해야 한다).
- gradient는 `.grad` 속성에 **누적**된다. 따라서 매 학습 스텝마다 `.grad`를 초기화해야 한다.

### 6. torch.no_grad()

추론(inference) 시에는 gradient 계산이 불필요하다. `torch.no_grad()` 컨텍스트를 사용하면 computational graph를 생성하지 않아 메모리와 연산을 절약한다.

```python
with torch.no_grad():
    pred = model(input_data)
```

VLA 모델처럼 파라미터가 큰 모델에서는 추론 시 이 절약 효과가 특히 크다.

---

## 연습 주제

1. 다양한 dtype과 shape으로 tensor를 생성한 뒤, `.shape`, `.dtype`, `.device`를 확인해 보라.
2. 두 tensor의 행렬 곱(`@`)을 수행하고, shape이 어떻게 결정되는지 확인해 보라.
3. `requires_grad=True`인 tensor로 간단한 수식(예: $y = x^2 + 3x$)을 만들고, `.backward()` 후 `.grad` 값을 수학적으로 검증해 보라.
4. CPU tensor와 CUDA tensor를 더하면 어떤 에러가 발생하는지 확인하고, 해결 방법을 적용해 보라.
5. `torch.no_grad()` 블록 안에서 연산한 결과 tensor에 `.grad_fn`이 있는지 확인해 보라.
6. gradient 누적 현상을 직접 관찰해 보라: `.backward()`를 두 번 호출한 뒤 `.grad` 값이 어떻게 변하는지 살펴보라.

---

## 다음 노트

[nn.Module](./02-nn-module.md) — tensor 연산을 체계적으로 묶어 "모델"이라는 단위로 관리하는 방법을 다룬다.
