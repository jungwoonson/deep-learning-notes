# OOP와 클래스 (Object-Oriented Programming)

> **시리즈**: Python Fundamentals 3/5
> **이전 노트**: [제어문과 함수](./02-control-flow-functions.md)
> **다음 노트**: [데코레이터와 제너레이터](./04-decorators-generators.md)

---

## 왜 필요한가 — VLA와의 연결

PyTorch에서 **모든 것이 클래스**다. VLA 모델을 이해하려면 반드시 OOP를 알아야 한다.

| Python OOP 개념 | PyTorch / VLA에서의 대응 |
|---|---|
| `class` 정의 | 모든 모델은 `class MyModel(nn.Module):`로 시작 |
| `__init__` | 모델의 레이어(layer) 구성 — 어떤 부품을 쓸지 정의 |
| 메서드(method) | `forward()` — 데이터가 모델을 통과하는 경로 정의 |
| 상속(inheritance) | `nn.Module`을 상속받아 커스텀 모델 생성 |
| `super()` | 부모 클래스 `nn.Module`의 초기화 호출 |
| `__len__`, `__getitem__` | `Dataset` 클래스 — 데이터셋의 크기와 개별 샘플 접근 |
| `__call__` | `model(input)` — 모델을 함수처럼 호출하는 원리 |
| `property` | 모델 파라미터 수, device 정보 등 계산된 속성 |

VLA 모델의 코드를 열면, 가장 먼저 마주하는 것이 `class VLAModel(nn.Module):`이다. 이 한 줄을 읽으려면 클래스, 상속, `__init__`, 메서드의 개념을 **모두** 알아야 한다.

---

## 핵심 개념

### 1. 클래스의 기본 구조

클래스는 **데이터(속성)와 동작(메서드)을 하나로 묶는 설계도**다.

```
class Robot:
    def __init__(self, name, num_joints):
        self.name = name                # 속성 (attribute)
        self.num_joints = num_joints
        self.joint_angles = [0.0] * num_joints

    def move(self, target_angles):      # 메서드 (method)
        self.joint_angles = target_angles

    def get_state(self):
        return self.joint_angles
```

- `__init__`: **생성자(constructor)**. 객체가 만들어질 때 자동 호출
- `self`: 자기 자신을 가리키는 참조. 모든 메서드의 첫 번째 인자
- 속성(attribute): `self.xxx`로 저장하는 데이터
- 메서드(method): 클래스에 속한 함수

#### 인스턴스 생성

```
robot = Robot("arm-01", 6)     # __init__ 자동 호출
robot.move([0.1, 0.2, ...])   # 메서드 호출
state = robot.get_state()      # 상태 조회
```

### 2. PyTorch nn.Module — 가장 중요한 클래스 패턴

PyTorch에서 모델을 정의하는 **표준 패턴**. 이 구조를 반드시 이해해야 한다.

```
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()                          # 부모 초기화 필수
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):                           # 데이터 흐름 정의
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

**패턴 분석**:
- `class SimpleModel(nn.Module):` — `nn.Module`을 **상속**
- `super().__init__()` — 부모 클래스의 초기화를 먼저 실행 (파라미터 추적 등 내부 설정)
- `self.layer1 = ...` — `__init__`에서 모델의 구성 요소(레이어)를 정의
- `forward(self, x)` — 입력 데이터 `x`가 모델을 어떻게 통과하는지 정의

### 3. 상속 (Inheritance)

**기존 클래스의 기능을 물려받아 확장**하는 것.

```
class BaseModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class VLAModel(BaseModel):                  # BaseModel을 상속
    def __init__(self, hidden_dim, num_actions):
        super().__init__(hidden_dim)        # 부모의 __init__ 호출
        self.vision_encoder = ...
        self.language_encoder = ...
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, image, text):
        ...
```

**상속의 의미**:
- `VLAModel`은 `BaseModel`의 모든 속성과 메서드를 갖고 있다
- `count_parameters()`를 다시 정의하지 않아도 사용 가능
- `super().__init__(hidden_dim)` — 부모의 생성자에 필요한 인자를 전달

#### super()의 역할

`super()`는 **부모 클래스를 참조**한다. 자식 클래스에서 부모의 메서드를 호출할 때 사용.

```
super().__init__()       # 부모의 __init__ 호출
super().forward(x)       # 부모의 forward 호출 (필요 시)
```

PyTorch에서 `super().__init__()`을 빠뜨리면 모델이 제대로 동작하지 않는다. 가장 흔한 실수 중 하나.

### 4. Property (프로퍼티)

메서드를 **속성처럼** 접근할 수 있게 만든다.

```
class Model(nn.Module):
    ...

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self):
        return next(self.parameters()).device
```

```
model = Model(...)
print(model.num_parameters)   # 함수 호출이 아니라 속성 접근처럼 사용
print(model.device)           # 괄호 없이 접근
```

**왜 필요한가**: 매번 계산이 필요하지만, 사용하는 쪽에서는 단순한 속성처럼 읽고 싶을 때. 모델의 파라미터 수, 현재 device 등이 대표적.

### 5. Dunder Methods (매직 메서드)

`__이름__` 형태의 특수 메서드. Python의 내장 연산자와 함수가 이 메서드를 호출한다.

#### `__len__` — len() 호출 시 실행

```
class ImageDataset:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)
```

```
dataset = ImageDataset(paths)
len(dataset)    # -> __len__() 호출 -> 이미지 개수 반환
```

#### `__getitem__` — 인덱싱([]) 시 실행

```
class ImageDataset:
    ...

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = load_image(path)
        label = self.labels[index]
        return image, label
```

```
dataset[0]      # -> __getitem__(0) 호출 -> (이미지, 라벨) 반환
dataset[42]     # -> __getitem__(42) 호출
```

**PyTorch Dataset의 핵심**: `__len__`과 `__getitem__`만 구현하면 DataLoader가 자동으로 배치를 만들고, 셔플하고, 병렬 로딩한다.

#### `__repr__` — 객체의 문자열 표현

```
class Model(nn.Module):
    ...

    def __repr__(self):
        return f"Model(hidden={self.hidden_dim}, layers={self.num_layers})"
```

```
print(model)    # -> __repr__() 호출 -> "Model(hidden=768, layers=12)"
```

PyTorch 모델을 `print()`하면 레이어 구조가 출력되는 이유가 바로 `__repr__` 덕분이다.

#### `__call__` — 객체를 함수처럼 호출

```
class Preprocessor:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        return resize(image, self.image_size)
```

```
preprocess = Preprocessor(224)
result = preprocess(image)    # -> __call__(image) 호출
```

**핵심**: PyTorch에서 `model(input)`이 동작하는 원리가 바로 이것이다. `nn.Module`은 `__call__` 내부에서 `forward()`를 호출하도록 구현되어 있다. 그래서 우리는 `forward()`만 정의하면 `model(x)` 형태로 사용할 수 있다.

---

## 기억할 멘탈 모델

```
PyTorch Dataset을 만드는 최소 패턴:

class VLADataset(Dataset):
    def __init__(self, data_path):
        self.samples = load(data_path)    # 데이터 로드

    def __len__(self):
        return len(self.samples)          # 전체 샘플 수

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = process_image(sample["image"])
        text = sample["instruction"]
        action = sample["action"]
        return image, text, action        # (이미지, 텍스트, 행동)


PyTorch Model을 만드는 최소 패턴:

class VLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()                # 부모 초기화 필수!
        self.vision = VisionEncoder(...)
        self.language = LanguageEncoder(...)
        self.action_head = ActionDecoder(...)

    def forward(self, image, text):
        vis_features = self.vision(image)       # __call__ -> forward
        lang_features = self.language(text)     # __call__ -> forward
        action = self.action_head(vis_features, lang_features)
        return action
```

이 두 패턴이 VLA를 포함한 **모든 PyTorch 프로젝트의 뼈대**다.

---

## 연습 주제

1. **클래스 설계 연습**: "로봇 팔" 클래스를 설계해보라. 어떤 속성(관절 수, 현재 위치, 그리퍼 상태)과 메서드(이동, 잡기, 놓기)가 필요한지 종이에 정리해보라.

2. **상속 관계 그리기**: `nn.Module` -> `BaseModel` -> `VLAModel` 상속 체인에서, 각 클래스가 어떤 기능을 담당하는지 다이어그램으로 그려보라.

3. **Dunder method 매핑**: `len(obj)`, `obj[i]`, `print(obj)`, `obj(x)` 각각이 어떤 dunder method를 호출하는지 표로 정리해보라.

4. **Dataset 설계**: VLA 학습 데이터셋을 위한 Dataset 클래스를 설계해보라. 하나의 샘플에 이미지, 텍스트 명령, 로봇 action이 포함되어야 한다. `__getitem__`이 무엇을 반환해야 하는지 생각해보라.

5. **super() 추적**: 3단계 상속(`A` -> `B` -> `C`)에서 `C`의 `__init__`이 호출될 때 `super().__init__()`의 호출 순서를 추적해보라.

---

## 다음 노트로

클래스를 이해했으면, 이제 클래스를 더 강력하게 만드는 **데코레이터**, 그리고 대용량 데이터를 효율적으로 처리하는 **제너레이터** 패턴을 배울 차례다.

> **다음**: [데코레이터와 제너레이터 (Decorators and Generators)](./04-decorators-generators.md)
