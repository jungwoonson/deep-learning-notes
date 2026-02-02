# 제어문과 함수 (Control Flow and Functions)

> **시리즈**: Python Fundamentals 2/5
> **이전 노트**: [기본 문법과 타입](./01-basics-syntax-types.md)
> **다음 노트**: [OOP와 클래스](./03-oop-classes.md)

---

## 왜 필요한가 — VLA와의 연결

딥러닝 코드는 본질적으로 **반복과 조건 분기의 조합**이다.

| Python 개념 | VLA에서 쓰이는 곳 |
|---|---|
| `for` 루프 | 학습 루프(training loop) — epoch, batch 순회 |
| `if/else` | 학습/추론 모드 분기, early stopping 조건 판단 |
| `while` | 로봇이 목표에 도달할 때까지 action 반복 실행 |
| 함수 정의 | 데이터 전처리, loss 계산, 평가 메트릭 등 재사용 가능한 단위 |
| `*args/**kwargs` | 유연한 함수 인터페이스 — PyTorch 모듈의 `forward()` 호출 |
| `lambda` | 짧은 변환 함수 — 데이터 augmentation 파이프라인 |
| `try/except` | 학습 중 에러 처리, 체크포인트 저장 실패 복구 |

VLA 모델의 학습 코드를 열어보면, 가장 바깥 구조가 `for epoch in range(num_epochs):` 루프이고, 그 안에 조건 분기와 함수 호출이 중첩되어 있다. 이 구조를 읽지 못하면 학습 파이프라인 전체를 이해할 수 없다.

---

## 핵심 개념

### 1. 조건문 (if / elif / else)

프로그램의 **분기(branching)** 를 담당한다.

```
# 학습률 스케줄링 (개념적 예시)
if epoch < warmup_steps:
    lr = base_lr * (epoch / warmup_steps)       # warmup 구간
elif epoch < total_epochs * 0.8:
    lr = base_lr                                 # 일정 유지
else:
    lr = base_lr * 0.1                           # 후반부 감소
```

**딥러닝에서의 핵심 패턴**:
- 학습(train) vs 추론(eval) 모드 분기
- 손실(loss) 값에 따른 early stopping
- GPU 사용 가능 여부에 따른 device 선택: `device = "cuda" if torch.cuda.is_available() else "cpu"`

#### Truthiness (참/거짓 판별)

Python에서는 `bool`이 아닌 값도 조건문에 쓸 수 있다.

- **Falsy** (거짓으로 취급): `0`, `0.0`, `""`, `[]`, `{}`, `None`, `False`
- **Truthy** (참으로 취급): 그 외 모든 값

```
# 흔한 패턴: 리스트가 비어있지 않으면 처리
if validation_results:
    compute_metrics(validation_results)
```

### 2. 반복문 (for / while)

#### for 루프 — 정해진 시퀀스를 순회

```
# 딥러닝의 가장 핵심적인 구조: 학습 루프
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

이 **이중 for 루프** 구조는 거의 모든 딥러닝 학습 코드에서 동일하다. 바깥 루프가 epoch, 안쪽 루프가 batch를 순회한다.

#### enumerate — 인덱스와 값을 동시에

```
for step, batch in enumerate(dataloader):
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}")
```

매 N 스텝마다 로그를 출력하는 패턴. `enumerate`가 `(인덱스, 값)` 쌍을 제공한다.

#### zip — 여러 시퀀스를 병렬로 순회

```
for prediction, target in zip(predictions, targets):
    error = prediction - target
```

#### while 루프 — 조건이 만족될 때까지 반복

```
# 로봇 제어: 목표에 도달할 때까지 action 실행
while distance_to_goal > threshold:
    action = model.predict(current_observation)
    execute(action)
    distance_to_goal = compute_distance()
```

VLA의 **추론(inference) 단계**에서 while 루프가 자주 사용된다. 로봇이 task를 완료할 때까지 관측-예측-실행을 반복한다.

#### break / continue

```
# early stopping: 성능이 개선되지 않으면 학습 중단
for epoch in range(max_epochs):
    val_loss = evaluate(model)
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        break  # 학습 조기 종료
```

### 3. 함수 (Functions)

코드의 **재사용 가능한 단위**. 딥러닝에서는 데이터 전처리, 모델 구성, 평가 등이 모두 함수로 분리된다.

```
def compute_accuracy(predictions, targets):
    """예측값과 정답을 비교하여 정확도를 계산한다."""
    correct = sum(p == t for p, t in zip(predictions, targets))
    total = len(targets)
    return correct / total
```

**함수 구성 요소**:
- `def` 키워드로 정의
- 파라미터(parameter): 입력을 받는 변수
- `return`: 결과를 돌려줌 (없으면 `None` 반환)
- docstring: 함수 설명 (큰따옴표 세 개)

#### 기본값 파라미터 (Default Parameters)

```
def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    ...
```

딥러닝 함수는 **수많은 hyperparameter**를 인자로 받는다. 기본값을 설정하면 자주 바꾸지 않는 값은 생략할 수 있다.

### 4. *args와 **kwargs

함수가 **가변 개수의 인자**를 받을 수 있게 해준다.

```
# *args: 위치 인자를 tuple로 받음
def log_metrics(*values):
    # values = (0.95, 0.87, 0.91) 처럼 tuple로 전달됨
    ...

# **kwargs: 키워드 인자를 dict로 받음
def create_model(**kwargs):
    # kwargs = {"hidden_size": 768, "num_layers": 12} 처럼 dict로 전달됨
    ...
```

**왜 중요한가**: PyTorch의 `nn.Module.forward()` 메서드를 오버라이드할 때, 부모 클래스의 `__init__`에 `*args, **kwargs`를 전달하는 패턴이 매우 흔하다. 이 문법을 모르면 PyTorch 코드를 읽을 수 없다.

### 5. Lambda 함수

이름 없는 **한 줄짜리 함수**. 간단한 변환에 사용한다.

```
# 정렬 기준 지정
checkpoints.sort(key=lambda x: x["step"])

# 데이터 변환
normalize = lambda x: (x - mean) / std
```

#### map / filter

```
# map: 모든 요소에 함수 적용
normalized = list(map(lambda x: x / 255.0, pixel_values))

# filter: 조건에 맞는 요소만 선택
valid_samples = list(filter(lambda s: s["quality"] > 0.5, dataset))
```

실무에서는 list comprehension이 더 선호되지만, 기존 코드베이스에서 map/filter를 자주 만나므로 읽을 수 있어야 한다.

### 6. 예외 처리 (try / except)

프로그램이 **오류로 중단되지 않도록** 보호한다.

```
# 체크포인트 저장 — 디스크 오류에도 학습이 중단되지 않도록
try:
    save_checkpoint(model, path)
except IOError:
    print("체크포인트 저장 실패, 학습은 계속...")

# 여러 예외 처리
try:
    data = load_data(path)
except FileNotFoundError:
    print("데이터 파일이 없습니다")
except ValueError:
    print("데이터 형식이 잘못되었습니다")
finally:
    cleanup()  # 성공/실패와 관계없이 항상 실행
```

**딥러닝에서의 핵심 용도**:
- 체크포인트 저장/로드 실패 처리
- 데이터 로딩 중 손상된 파일 건너뛰기
- GPU 메모리 부족(OOM) 시 배치 크기 줄이기

---

## 기억할 멘탈 모델

```
VLA 학습 코드의 전체 구조:

def train(config):                          # 함수
    model = build_model(**config)           # **kwargs

    for epoch in range(config["epochs"]):   # 바깥 for 루프
        for batch in dataloader:            # 안쪽 for 루프
            try:                            # 예외 처리
                loss = model(batch)
                loss.backward()
                optimizer.step()
            except RuntimeError:
                handle_oom()

            if step % log_interval == 0:    # 조건문
                log_metrics(loss)

        if should_stop_early(val_loss):     # 조건문 + 함수
            break                           # 루프 탈출
```

이 구조가 사실상 **모든 딥러닝 학습 코드의 골격**이다.

---

## 연습 주제

1. **학습 루프 설계**: epoch, batch 이중 루프의 구조를 종이에 그려보라. 어느 시점에서 loss를 계산하고, 어느 시점에서 로그를 남기고, 어느 시점에서 모델을 저장하는 것이 적절한지 생각해보라.

2. **Early stopping 로직**: patience=5일 때 early stopping이 어떻게 동작하는지 흐름도(flowchart)를 그려보라. 어떤 변수가 필요한가?

3. **함수 분리 연습**: 학습 루프의 어떤 부분을 별도 함수로 분리하면 좋을지 생각해보라. (힌트: `train_one_epoch()`, `evaluate()`, `save_checkpoint()`)

4. **kwargs 읽기 연습**: `def __init__(self, hidden_size=768, num_heads=12, **kwargs)` 같은 함수 시그니처를 보고, 어떤 인자가 필수이고 어떤 것이 선택적인지 구분해보라.

5. **예외 상황 열거**: 딥러닝 학습 중 발생할 수 있는 오류들을 나열하고, 각각에 대해 어떤 except 처리가 적절한지 생각해보라.

---
