# 파이썬 생태계 (Python Ecosystem)

> **시리즈**: Python Fundamentals 5/5
> **이전 노트**: [데코레이터와 제너레이터](./04-decorators-generators.md)
> **다음 폴더**: [NumPy와 데이터 처리](../03-numpy-data/)

---

## 왜 필요한가 — VLA와의 연결

VLA 모델을 학습하거나 실행하려면 Python 코드 작성 능력만으로는 부족하다. **수십 개의 라이브러리**를 설치하고, **환경 충돌 없이** 관리하며, **실험을 기록**할 수 있어야 한다.

| 생태계 도구 | VLA 프로젝트에서의 역할 |
|---|---|
| pip / conda | PyTorch, Transformers, 로봇 제어 라이브러리 설치 |
| 가상환경 (venv/conda) | 프로젝트별 의존성 격리 — PyTorch 버전 충돌 방지 |
| Jupyter Notebook | 데이터 탐색, 모델 실험, 결과 시각화 |
| 프로젝트 구조 | 코드 재사용, 팀 협업, 재현 가능한 실험 |
| git | 코드 버전 관리, 실험 추적, 협업 |

실제 VLA 연구 환경에서는 코드 작성보다 **환경 설정에 더 많은 시간**을 쓰는 경우가 흔하다. 특히 GPU 드라이버, CUDA 버전, PyTorch 버전의 호환성 문제는 모든 딥러닝 엔지니어의 숙명이다.

---

## 핵심 개념

### 1. pip — Python 패키지 관리자

pip는 **Python Package Index(PyPI)** 에서 라이브러리를 설치하는 도구다.

#### 기본 명령어

```
pip install numpy                  # 최신 버전 설치
pip install torch==2.1.0           # 특정 버전 설치
pip install transformers>=4.30     # 최소 버전 지정
pip install -r requirements.txt    # 파일에 명시된 패키지 일괄 설치
pip uninstall numpy                # 제거
pip list                           # 설치된 패키지 목록
pip freeze > requirements.txt      # 현재 환경의 패키지를 파일로 저장
```

#### requirements.txt

프로젝트에 필요한 패키지를 명시하는 파일. **재현 가능한 환경**의 핵심.

```
# requirements.txt 예시 (VLA 프로젝트)
torch==2.1.0
torchvision==0.16.0
transformers>=4.35.0
numpy>=1.24.0
pillow>=9.0
wandb>=0.15.0
```

다른 사람이 이 프로젝트를 받으면 `pip install -r requirements.txt` 한 줄로 동일한 환경을 구성할 수 있다.

### 2. 가상환경 (Virtual Environment)

**프로젝트마다 독립된 Python 환경**을 만드는 것. 왜 필요한가?

- 프로젝트 A는 PyTorch 2.0 필요, 프로젝트 B는 PyTorch 1.13 필요
- 가상환경 없이 전역에 설치하면 **버전 충돌**이 발생
- 가상환경을 사용하면 각 프로젝트가 독립된 패키지 세트를 가짐

#### venv — Python 내장 가상환경

```
# 가상환경 생성
python -m venv myenv

# 활성화 (macOS/Linux)
source myenv/bin/activate

# 활성화 (Windows)
myenv\Scripts\activate

# 활성화된 상태에서 패키지 설치 — 이 환경에만 설치됨
pip install torch

# 비활성화
deactivate
```

#### conda — 더 강력한 환경 관리

conda는 Python 패키지뿐 아니라 **시스템 레벨 의존성**(CUDA, cuDNN 등)까지 관리할 수 있다. 딥러닝에서는 venv보다 conda가 더 많이 사용된다.

```
# 환경 생성 (Python 버전 지정 가능)
conda create -n vla-project python=3.10

# 환경 활성화
conda activate vla-project

# 패키지 설치 (conda 채널에서)
conda install pytorch torchvision -c pytorch

# pip와 혼용 가능
pip install transformers

# 환경 목록 확인
conda env list

# 환경 내보내기 (재현용)
conda env export > environment.yml

# 환경 복원
conda env create -f environment.yml
```

**venv vs conda 선택 기준**:
- 간단한 프로젝트, Python 패키지만 필요 -> `venv`
- GPU 사용, CUDA 필요, 복잡한 의존성 -> `conda` (권장)
- 대부분의 딥러닝 프로젝트 -> `conda`

### 3. Jupyter Notebook

**코드를 셀(cell) 단위로 실행**하고 결과를 즉시 확인할 수 있는 대화형 환경.

#### 왜 딥러닝에서 중요한가

- 데이터 탐색: 이미지를 직접 보고, 분포를 그래프로 확인
- 실험: 모델의 하이퍼파라미터를 바꿔가며 결과 비교
- 시각화: 학습 곡선, attention map, 로봇 trajectory를 인라인으로 표시
- 문서화: 코드와 설명을 한 파일에 함께 작성

#### 셀 타입

- **Code cell**: Python 코드 실행
- **Markdown cell**: 설명 텍스트 (제목, 수식, 이미지 등)

#### 설치와 실행

```
pip install jupyter
jupyter notebook        # 브라우저에서 열림

# 또는 JupyterLab (더 현대적인 인터페이스)
pip install jupyterlab
jupyter lab
```

#### Jupyter 사용 시 주의점

- 셀 실행 순서가 중요하다 — 위에서 아래로 순차적으로 실행할 것
- 전역 상태가 공유된다 — 한 셀에서 변수를 바꾸면 다른 셀에 영향
- **최종 코드는 `.py` 파일로 정리**할 것 — notebook은 실험용, 프로덕션은 스크립트
- GPU 메모리를 잡고 있을 수 있다 — 커널 재시작으로 해제

### 4. 프로젝트 구조 (Project Structure)

딥러닝 프로젝트는 대체로 아래와 같은 구조를 따른다.

```
vla-project/
    README.md               # 프로젝트 설명
    requirements.txt         # 의존성 목록
    setup.py                 # 패키지 설치 설정 (선택)

    configs/                 # 실험 설정 파일
        base.yaml
        vla_large.yaml

    data/                    # 데이터 (보통 .gitignore에 추가)
        raw/
        processed/

    src/                     # 소스 코드
        __init__.py
        model/
            __init__.py
            vision.py        # Vision encoder
            language.py      # Language encoder
            action.py        # Action decoder
            vla.py           # 통합 VLA 모델
        data/
            __init__.py
            dataset.py       # Dataset 클래스
            transforms.py    # 데이터 전처리
        training/
            __init__.py
            trainer.py       # 학습 루프
            optimizer.py     # 옵티마이저 설정
        utils/
            __init__.py
            logging.py
            metrics.py

    scripts/                 # 실행 스크립트
        train.py
        evaluate.py
        inference.py

    notebooks/               # 실험/탐색용 Jupyter
        01_data_exploration.ipynb
        02_model_testing.ipynb

    tests/                   # 테스트 코드
        test_model.py
        test_data.py
```

**핵심 원칙**:
- `src/`에 재사용 가능한 모듈을 넣고, `scripts/`에서 이를 조합하여 실행
- `configs/`로 실험 설정을 코드에서 분리 (hardcoding 금지)
- `data/`는 git으로 추적하지 않음 (용량이 크므로)
- `__init__.py` 파일이 있어야 Python이 해당 디렉토리를 패키지로 인식

#### __init__.py의 역할

```
# src/model/__init__.py
from .vla import VLAModel
from .vision import VisionEncoder
```

이 파일이 있으면 다른 곳에서 이렇게 import할 수 있다:

```
from src.model import VLAModel
```

`__init__.py`가 없으면 `from src.model import ...` 구문이 동작하지 않는다.

### 5. Git 기초 — 버전 관리

Git은 **코드의 변경 이력을 추적**하는 도구. 딥러닝에서는 실험 코드의 버전 관리에 필수적이다.

#### 핵심 개념

- **Repository (저장소)**: 프로젝트의 전체 이력을 담은 폴더
- **Commit**: 특정 시점의 코드 스냅샷 — "이 시점에서 모델이 잘 동작했음"
- **Branch**: 독립적인 작업 흐름 — "새로운 loss 함수를 실험할 때 별도 브랜치에서"
- **Remote**: GitHub 등 원격 저장소 — 백업, 협업, 코드 공유

#### 기본 워크플로우

```
# 저장소 초기화
git init

# 현재 상태 확인
git status

# 변경사항 스테이징 (커밋 준비)
git add train.py
git add src/model/vla.py

# 커밋 (스냅샷 저장)
git commit -m "VLA 모델 학습 코드 추가"

# 변경 이력 확인
git log --oneline

# 원격 저장소에 업로드
git push origin main
```

#### .gitignore — 추적하지 않을 파일 지정

```
# .gitignore 예시 (딥러닝 프로젝트)
data/                    # 대용량 데이터
*.pt                     # 모델 체크포인트 (수 GB)
*.pth
wandb/                   # 실험 로그
__pycache__/             # Python 캐시
.ipynb_checkpoints/      # Jupyter 캐시
*.pyc
.env                     # 환경 변수 (API 키 등)
```

모델 가중치 파일(수 GB)과 데이터셋을 git에 넣으면 저장소가 망가진다. 반드시 `.gitignore`에 추가할 것.

#### 딥러닝에서 git을 쓰는 이유

- "3일 전 코드가 더 잘 동작했는데..." -> `git log`로 되돌아가기
- "이 실험은 어떤 코드로 했지?" -> commit hash로 추적
- 팀원과 모델 코드 공유 -> `git push` / `git pull`
- 논문 재현 -> 특정 commit을 checkout하면 정확히 그 시점의 코드

---

## 기억할 멘탈 모델

```
VLA 프로젝트 시작부터 학습까지의 흐름:

1. 환경 구성
   conda create -n vla python=3.10
   conda activate vla
   pip install -r requirements.txt

2. 프로젝트 구조 생성
   mkdir -p src/{model,data,training} scripts configs

3. 코드 작성
   src/model/vla.py         (모델 정의 — class, nn.Module)
   src/data/dataset.py      (데이터셋 — __len__, __getitem__)
   scripts/train.py         (학습 스크립트 — for loop)
   configs/base.yaml        (하이퍼파라미터 — dict)

4. 실험
   jupyter lab               (데이터 탐색, 모델 테스트)

5. 학습 실행
   python scripts/train.py --config configs/base.yaml

6. 버전 관리
   git add . && git commit -m "baseline VLA 학습 완료"
```

---

## 연습 주제

1. **환경 구성 실습**: conda로 새 환경을 만들고, PyTorch와 NumPy를 설치한 뒤, Python에서 import가 되는지 확인해보라. 환경을 삭제하고 다시 만들어보라.

2. **requirements.txt 작성**: VLA 프로젝트에 필요할 것 같은 패키지 목록을 작성해보라. 각 패키지가 왜 필요한지 한 줄 주석으로 설명을 달아보라.

3. **프로젝트 구조 설계**: 위의 프로젝트 구조 템플릿을 참고하여, 빈 디렉토리와 `__init__.py` 파일을 직접 만들어보라. 각 디렉토리의 역할을 설명할 수 있는가?

4. **git 워크플로우 연습**: 간단한 Python 파일을 만들고, git init -> add -> commit -> 수정 -> add -> commit의 과정을 직접 수행해보라. `git log`로 이력을 확인해보라.

5. **Jupyter vs Script 판단**: "데이터 분포 확인", "모델 학습 실행", "결과 시각화", "배포용 추론 코드" 각각을 Jupyter와 .py 스크립트 중 어디서 작성하는 것이 적절한지 판단하고 이유를 설명해보라.

---

## 다음 단계로

Python 기초를 마쳤다. 이제 **데이터를 실제로 다루는 도구**를 배울 차례다. NumPy는 Python에서 수치 계산을 수행하는 핵심 라이브러리이며, PyTorch 텐서의 직접적인 조상이다.

> **다음 폴더**: [NumPy와 데이터 처리 (numpy-data)](../03-numpy-data/)
