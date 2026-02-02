# Dataset과 DataLoader

## 왜 필요한가 — VLA와의 연결

VLA 모델의 학습 데이터는 단순하지 않다. 하나의 샘플이 이미지(고해상도 RGB), 자연어 명령(가변 길이 텍스트), 로봇 행동 시퀀스(연속 값 벡터)를 동시에 포함한다. 이처럼 이질적인 데이터를 하나의 파이프라인으로 로드하고, 배치로 묶고, 전처리를 적용하려면 체계적인 데이터 추상화가 필요하다.

PyTorch의 `Dataset`은 "데이터 하나를 어떻게 꺼내는가"를 정의하고, `DataLoader`는 "여러 데이터를 어떻게 배치로 묶어 모델에 전달하는가"를 처리한다. VLA 학습에서 수십만 개의 로봇 에피소드 데이터를 효율적으로 순회하려면 이 두 클래스를 제대로 이해해야 한다.

---

## 핵심 개념

### 1. torch.utils.data.Dataset

`Dataset`을 상속하고 두 가지 메서드를 구현하면 커스텀 데이터셋을 만들 수 있다.

- **`__len__`**: 전체 데이터 수를 반환한다.
- **`__getitem__`**: 인덱스를 받아 해당 샘플을 반환한다.

```python
class RobotDataset(Dataset):
    def __init__(self, episodes, transform=None):
        self.episodes = episodes
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        sample = self.episodes[idx]
        image = sample["image"]
        if self.transform:
            image = self.transform(image)
        return image, sample["text"], sample["action"]
```

`__getitem__`이 반환하는 것은 tuple, dict 등 자유로운 형태가 가능하다. VLA 데이터셋에서는 dict 형태(`{"image": ..., "text": ..., "action": ...}`)를 자주 사용한다.

### 2. DataLoader

`DataLoader`는 `Dataset`을 감싸서 배치 생성, 셔플링, 병렬 로딩 등을 자동으로 처리한다.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

**주요 파라미터**:

| 파라미터 | 역할 |
|----------|------|
| `batch_size` | 한 번에 묶는 샘플 수 |
| `shuffle` | 에포크마다 데이터 순서를 무작위로 섞을지 여부 (학습 시 `True`, 검증 시 `False`) |
| `num_workers` | 데이터 로딩에 사용하는 서브 프로세스 수. 0이면 메인 프로세스에서 로딩 |
| `pin_memory` | `True`로 설정하면 CPU → GPU 전송 속도가 향상됨 |
| `drop_last` | 마지막 불완전한 배치를 버릴지 여부 |

### 3. torchvision.transforms

이미지 전처리 파이프라인을 함수 조합(composition)으로 구성한다.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

- `ToTensor()`: PIL 이미지 또는 NumPy 배열을 `(C, H, W)` 형태의 float tensor로 변환한다.
- `Normalize()`: 채널별 평균과 표준편차로 정규화한다. 사전 학습된 vision encoder를 사용할 때는 해당 모델의 정규화 값을 맞춰야 한다.
- 데이터 증강(augmentation): `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter` 등을 학습 시에만 적용한다.

VLA에서 vision encoder가 기대하는 입력 해상도와 정규화 값이 다르면 성능이 크게 저하되므로, 전처리를 정확히 맞추는 것이 중요하다.

### 4. Custom Collate Function

`DataLoader`는 기본적으로 각 샘플의 같은 위치 요소끼리 `torch.stack`으로 배치를 구성한다. 하지만 가변 길이 텍스트나 서로 다른 크기의 데이터가 섞여 있으면 기본 collate가 실패한다.

이 경우 **`collate_fn`**을 직접 정의한다.

```python
def custom_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]  # 가변 길이: 리스트로 유지
    actions = torch.stack([item["action"] for item in batch])
    return {"image": images, "text": texts, "action": actions}

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
```

VLA 학습에서는 텍스트 길이가 제각각이므로, collate 함수 내에서 padding을 적용하거나 tokenizer의 `pad` 기능을 활용하는 경우가 많다.

### 5. IterableDataset

매우 큰 데이터(수 TB의 로봇 데모 데이터 등)는 전체를 메모리에 올릴 수 없다. 이때 `IterableDataset`을 사용하면 데이터를 스트리밍 방식으로 읽을 수 있다. `__getitem__` 대신 `__iter__`를 구현한다.

---

## 연습 주제

1. 간단한 숫자 데이터로 `Dataset`을 만들고, `DataLoader`를 통해 배치 단위로 순회해 보라.
2. `shuffle=True`와 `shuffle=False`일 때 배치 구성이 어떻게 달라지는지 비교해 보라.
3. 이미지 파일을 읽어 `transforms.Compose`를 적용하는 Dataset을 만들어 보라.
4. 가변 길이 리스트를 포함하는 Dataset에서 기본 `DataLoader`가 실패하는 상황을 확인하고, `collate_fn`으로 해결해 보라.
5. `num_workers`를 0, 2, 4로 바꿔 가며 데이터 로딩 속도 차이를 측정해 보라.
6. `drop_last=True`와 `False`일 때 전체 에포크에서 처리되는 샘플 수의 차이를 확인해 보라.

---
