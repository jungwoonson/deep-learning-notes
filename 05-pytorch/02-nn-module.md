# nn.Module

## 왜 필요한가 — VLA와의 연결

VLA 모델은 단일 레이어가 아니라, vision encoder, language model, action head 등 여러 서브 모듈이 계층적으로 결합된 구조다. PyTorch에서 이런 복합 구조를 체계적으로 정의하고 관리하는 방법이 `nn.Module`이다.

`nn.Module`을 상속하면 파라미터 등록, forward 연산 정의, 학습/추론 모드 전환, 체크포인트 저장/로드가 모두 일관된 인터페이스로 동작한다. VLA 논문에서 "we freeze the vision encoder and fine-tune the language model" 같은 표현이 가능한 이유도, 각 구성 요소가 `nn.Module`로 분리되어 있기 때문이다.

---

## 핵심 개념

### 1. nn.Module 기본 구조

`nn.Module`을 상속한 클래스는 두 가지 메서드를 반드시 정의한다.

- **`__init__`**: 레이어(파라미터를 가진 연산)를 선언한다.
- **`forward`**: 입력 데이터가 레이어를 통과하는 순서를 정의한다.

```python
class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))
```

호출은 `model(x)` 형태로 하며, 이것이 내부적으로 `forward(x)`를 실행한다. 절대 `model.forward(x)`를 직접 호출하지 않는다(hook 등 내부 메커니즘이 우회되기 때문).

### 2. 자주 쓰는 내장 레이어

| 레이어 | 역할 | VLA에서의 용도 |
|--------|------|----------------|
| `nn.Linear(in, out)` | 선형 변환 $y = xW^T + b$ | projection head, MLP |
| `nn.ReLU()` | 활성 함수 | 비선형성 추가 |
| `nn.Embedding(num, dim)` | 정수 인덱스를 벡터로 변환 | 토큰 임베딩 |
| `nn.LayerNorm(dim)` | 레이어 정규화 | Transformer 블록 내부 |
| `nn.Dropout(p)` | 학습 시 무작위 뉴런 비활성화 | 과적합 방지 |
| `nn.Conv2d(in_ch, out_ch, k)` | 2D 합성곱 | vision encoder |
| `nn.MultiheadAttention(dim, heads)` | 멀티헤드 어텐션 | Transformer 핵심 |

### 3. nn.Sequential

레이어를 순차적으로 쌓을 때 간결하게 표현할 수 있다.

```python
mlp = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
)
```

단, 분기(branch)나 잔차 연결(residual connection)이 있는 구조에는 적합하지 않다. 그런 경우 `forward` 메서드에서 직접 흐름을 제어한다.

### 4. parameters()와 named_parameters()

`nn.Module`에 등록된 모든 학습 가능한 파라미터를 순회할 수 있다.

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

이 메서드는 optimizer에 파라미터를 전달할 때, 특정 레이어를 freeze할 때, 총 파라미터 수를 계산할 때 사용된다. VLA에서 vision encoder만 freeze하려면 해당 서브 모듈의 파라미터에 대해 `param.requires_grad = False`로 설정한다.

### 5. model.train()과 model.eval()

- **`model.train()`**: Dropout 활성화, BatchNorm이 배치 통계 사용 등 학습 모드 동작.
- **`model.eval()`**: Dropout 비활성화, BatchNorm이 누적 통계 사용 등 추론 모드 동작.

이 전환을 빠뜨리면 추론 결과가 비결정적(non-deterministic)으로 변하거나, 학습 성능이 떨어질 수 있다. 특히 VLA처럼 사전 학습된 서브 모듈을 조합하는 경우, 각 모듈의 train/eval 상태를 정확히 관리해야 한다.

### 6. 서브 모듈 등록

`nn.Module` 내부에 다른 `nn.Module`을 속성으로 할당하면 자동으로 서브 모듈로 등록된다.

```python
class VLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.language_model = LanguageModel()
        self.action_head = ActionHead()
```

이렇게 하면 `model.parameters()`가 세 서브 모듈의 파라미터를 모두 포함하며, `model.state_dict()`도 계층적 키(예: `vision_encoder.conv1.weight`)로 저장된다.

---

## 연습 주제

1. `nn.Module`을 상속하여 2-layer MLP를 직접 정의하고, 임의의 입력을 통과시켜 출력 shape을 확인해 보라.
2. `parameters()`로 모델의 총 파라미터 수를 계산해 보라.
3. 특정 레이어의 `requires_grad`를 `False`로 설정한 뒤, `parameters()` 순회 시 해당 파라미터가 여전히 포함되는지 확인해 보라 (포함되지만 gradient는 계산되지 않는다).
4. `nn.Sequential`로 같은 MLP를 만들고, 직접 정의한 버전과 `state_dict()`의 키 구조를 비교해 보라.
5. `model.train()`과 `model.eval()`을 전환하면서 `Dropout` 레이어의 출력이 어떻게 달라지는지 관찰해 보라.
6. 두 개의 서브 모듈을 가진 모델을 만들고, `named_parameters()`로 키 이름의 계층 구조를 확인해 보라.

---

## 다음 노트

[Dataset과 DataLoader](./03-datasets-dataloaders.md) — 모델에 공급할 데이터를 효율적으로 준비하고 배치 단위로 전달하는 방법을 다룬다.
