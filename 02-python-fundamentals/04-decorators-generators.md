# 데코레이터와 제너레이터 (Decorators and Generators)

> **시리즈**: Python Fundamentals 4/5
> **이전 노트**: [OOP와 클래스](./03-oop-classes.md)
> **다음 노트**: [파이썬 생태계](./05-ecosystem-pip-venv.md)

---

## 왜 필요한가 — VLA와의 연결

데코레이터와 제너레이터는 Python의 **고급 기능**이지만, PyTorch 코드에서는 **기본처럼** 사용된다. 이 개념을 모르면 PyTorch 코드를 읽을 수 없다.

| Python 개념 | VLA / PyTorch에서의 대응 |
|---|---|
| `@property` | 모델 속성을 메서드로 계산하되, 속성처럼 접근 |
| `@staticmethod` | 인스턴스 없이 사용하는 유틸리티 함수 |
| `@torch.no_grad()` | 추론 시 gradient 계산 비활성화 — 메모리 절약 |
| `@torch.compile` | 모델 최적화 컴파일 (PyTorch 2.0+) |
| `yield` / generator | DataLoader가 배치를 하나씩 생산 — 전체를 메모리에 올리지 않음 |
| `with` (context manager) | GPU 메모리 관리, 파일 I/O, gradient 제어 |

VLA 모델은 이미지 + 텍스트 + action이라는 **대용량 데이터**를 다룬다. 제너레이터 없이는 메모리가 부족하고, 데코레이터 없이는 코드가 반복적이고 장황해진다.

---

## 핵심 개념

### 1. 데코레이터 (Decorators) — 함수를 감싸는 함수

데코레이터는 **기존 함수의 동작을 변경하거나 확장**하는 패턴이다. `@` 기호를 사용한다.

#### 데코레이터의 원리

```
# 데코레이터는 사실 이런 의미다:
@decorator
def my_function():
    ...

# 위 코드는 아래와 동일하다:
my_function = decorator(my_function)
```

데코레이터는 함수를 입력으로 받아, **새로운(감싸진) 함수를 반환**한다. 원래 함수의 코드를 수정하지 않고 기능을 추가할 수 있다.

#### 개념적 예시 — 실행 시간 측정

```
@measure_time
def train_one_epoch(model, dataloader):
    ...

# train_one_epoch()를 호출하면
# 자동으로 실행 시간이 측정되어 출력된다
# 함수 내부 코드는 전혀 수정하지 않았다
```

### 2. PyTorch에서 자주 만나는 데코레이터들

#### @property — 메서드를 속성처럼

이전 노트에서 다뤘지만 데코레이터 관점에서 다시 보자.

```
class Model(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device
```

`@property`는 Python 내장 데코레이터다. `device()`라는 메서드를 `model.device`처럼 괄호 없이 접근할 수 있게 만든다.

#### @staticmethod — self 없는 메서드

```
class DataProcessor:
    @staticmethod
    def normalize(image):
        return image / 255.0
```

- `self`를 받지 않으므로 인스턴스 없이 호출 가능: `DataProcessor.normalize(img)`
- 클래스에 논리적으로 속하지만 인스턴스 상태가 필요 없는 유틸리티 함수에 사용
- PyTorch Dataset의 전처리 함수에서 자주 보인다

#### @classmethod — 클래스 자체를 받는 메서드

```
class VLAModel(nn.Module):
    @classmethod
    def from_pretrained(cls, model_path):
        config = load_config(model_path)
        model = cls(config)              # cls = VLAModel 클래스 자체
        weights = load_weights(model_path)
        model.load_state_dict(weights)
        return model
```

```
model = VLAModel.from_pretrained("path/to/model")
```

**팩토리 메서드(factory method)** 패턴. Hugging Face의 `AutoModel.from_pretrained()`가 바로 이 패턴이다. VLA 모델을 사전학습된 가중치에서 로드할 때 반드시 만나게 된다.

#### @torch.no_grad() — gradient 계산 비활성화

```
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    for batch in dataloader:
        predictions = model(batch)
        ...
```

- 추론(inference) 시에는 gradient가 필요 없다
- gradient를 계산하지 않으면 **메모리 사용량이 크게 감소**하고 **속도가 빨라진다**
- VLA 모델은 크기가 크므로 이 최적화가 필수적

#### @torch.compile — 모델 최적화 (PyTorch 2.0+)

```
@torch.compile
def forward_pass(model, inputs):
    return model(inputs)
```

Python 코드를 최적화된 커널로 컴파일한다. 데코레이터 한 줄로 성능 향상을 얻는다.

### 3. 제너레이터 (Generators) — 데이터를 하나씩 생산

제너레이터는 **모든 데이터를 한 번에 메모리에 올리지 않고, 필요할 때마다 하나씩 생성**하는 함수다.

#### 일반 함수 vs 제너레이터

```
# 일반 함수: 모든 결과를 리스트로 한번에 반환
def get_all_batches(dataset, batch_size):
    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i+batch_size])
    return batches          # 전체가 메모리에 올라감

# 제너레이터: yield로 하나씩 반환
def get_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]   # 하나 생산하고 일시 정지
```

**핵심 차이**: `return`은 함수를 종료하지만, `yield`는 **값을 내보내고 일시 정지**한다. 다음 값을 요청하면 중단된 지점부터 다시 실행된다.

#### 왜 중요한가 — 메모리 효율

VLA 학습 데이터를 생각해보자:
- 이미지 수만 장 (각각 수 MB)
- 텍스트 명령 수만 개
- Action 시퀀스 수만 개

이 모든 데이터를 **한 번에 메모리에 올릴 수 없다**. 제너레이터 패턴을 사용하면 현재 배치(batch)만 메모리에 올리고, 처리가 끝나면 다음 배치를 생성한다.

```
# PyTorch DataLoader가 내부적으로 하는 일 (개념적)
def dataloader(dataset, batch_size):
    indices = list(range(len(dataset)))
    shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = [dataset[idx] for idx in batch_indices]
        yield collate(batch)     # 배치 하나를 생산하고 대기
```

```
# 사용하는 쪽 — for 루프가 자동으로 yield된 값을 하나씩 받음
for batch in dataloader(dataset, batch_size=32):
    loss = model(batch)
    ...
```

#### 제너레이터 표현식

리스트 컴프리헨션과 비슷하지만 `[]` 대신 `()`를 사용한다.

```
# 리스트 컴프리헨션 — 전체를 메모리에 생성
squares_list = [x**2 for x in range(1000000)]

# 제너레이터 표현식 — 하나씩 생산
squares_gen = (x**2 for x in range(1000000))
```

첫 번째는 100만 개의 값을 즉시 메모리에 만든다. 두 번째는 요청할 때마다 하나씩 계산한다.

### 4. Context Manager (컨텍스트 매니저) — with 문

**자원의 획득과 해제를 자동으로 관리**하는 패턴.

#### 기본 패턴 — 파일 처리

```
# with 문 사용 — 블록이 끝나면 자동으로 파일을 닫음
with open("config.json", "r") as f:
    config = json.load(f)
# 여기서는 f가 이미 닫힌 상태

# with 없이 — 직접 닫아야 하고, 에러 시 닫히지 않을 수 있음
f = open("config.json", "r")
config = json.load(f)
f.close()    # 에러가 나면 이 줄이 실행되지 않을 수 있다
```

#### PyTorch에서의 context manager

```
# gradient 계산 비활성화 (추론 시)
with torch.no_grad():
    predictions = model(input_data)

# 자동 혼합 정밀도 (Automatic Mixed Precision)
with torch.cuda.amp.autocast():
    output = model(input_data)
    loss = criterion(output, target)

# 모델 가중치 저장
with open("model.pt", "wb") as f:
    torch.save(model.state_dict(), f)
```

**`torch.no_grad()`의 두 가지 사용법**:
1. **데코레이터로**: `@torch.no_grad()` — 함수 전체에 적용
2. **컨텍스트 매니저로**: `with torch.no_grad():` — 특정 블록에만 적용

둘 다 자주 사용되므로 양쪽 모두 읽을 수 있어야 한다.

#### context manager의 원리

```
class NoGradContext:
    def __enter__(self):
        # with 블록 진입 시 실행: gradient 계산 비활성화
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        # with 블록 탈출 시 실행: 원래 상태로 복원
        # 에러가 발생해도 반드시 실행됨
        ...
```

`__enter__`와 `__exit__` dunder method로 구현된다. `__exit__`이 **에러 발생 여부와 관계없이** 항상 호출되므로, 자원 누수(resource leak)를 방지할 수 있다.

---

## 기억할 멘탈 모델

```
VLA 추론 파이프라인에서의 패턴 조합:

class VLAModel(nn.Module):
    @classmethod
    def from_pretrained(cls, path):      # 데코레이터: classmethod
        ...

    @property                             # 데코레이터: property
    def device(self):
        return next(self.parameters()).device

    def forward(self, image, text):
        ...

# 추론
model = VLAModel.from_pretrained("vla-7b")

with torch.no_grad():                    # context manager
    for batch in dataloader:             # dataloader = generator 패턴
        action = model(batch)
        execute_action(action)
```

이 짧은 코드에 데코레이터, 제너레이터, context manager가 **모두** 등장한다.

---

## 연습 주제

1. **데코레이터 분류**: PyTorch 코드에서 자주 보이는 `@property`, `@staticmethod`, `@classmethod`, `@torch.no_grad()`, `@torch.compile`을 각각 언제 사용하는지 표로 정리해보라.

2. **yield 추적**: 제너레이터 함수가 `yield`를 만나면 무슨 일이 일어나는지, 다음 `next()` 호출 시 어디서부터 재개되는지 흐름을 그려보라.

3. **메모리 비교 사고 실험**: 10만 장의 이미지를 처리할 때, (a) 전체를 리스트로 로드하는 경우와 (b) 제너레이터로 하나씩 로드하는 경우, 메모리 사용량이 어떻게 다른지 생각해보라.

4. **with 문 설계**: 학습 시작 시 타이머를 시작하고, 학습 종료(정상/에러 모두) 시 총 소요 시간을 출력하는 context manager를 개념적으로 설계해보라. `__enter__`와 `__exit__`에 각각 무엇을 넣을 것인가?

5. **@torch.no_grad() 사용 판단**: 학습 루프, 검증 루프, 추론 루프 각각에서 `torch.no_grad()`를 써야 하는지, 쓰지 말아야 하는지 판단하고 이유를 설명해보라.

---

## 다음 노트로

Python 언어 자체의 핵심 기능은 여기까지다. 이제 실제로 프로젝트를 만들고 실행하기 위한 **생태계 도구**를 배울 차례다. 패키지 관리, 가상환경, Jupyter, 프로젝트 구조.

> **다음**: [파이썬 생태계 (Python Ecosystem)](./05-ecosystem-pip-venv.md)
