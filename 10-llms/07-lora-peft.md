# LoRA와 PEFT

## VLA와의 연결

**OpenVLA는 새로운 로봇이나 태스크에 적응할 때 LoRA를 사용한다.** 7B 파라미터 모델 전체를 파인튜닝하는 대신, **전체의 약 1.4%에 해당하는 파라미터만 학습**한다. 이것이 가능한 이유와 작동 원리를 이 노트에서 다룬다.

구체적으로:
- OpenVLA 7B 전체 파라미터: 약 7.6B
- LoRA 학습 파라미터: 약 107M (1.4%)
- LoRA rank: r = 32
- 적용 대상: attention 레이어의 모든 projection (Q, K, V, O)

이 효율성 덕분에 단일 GPU에서도 로봇별 파인튜닝이 가능해진다.

---

## 핵심 개념

### 1. Full Fine-tuning의 비용 문제

7B 모델을 full fine-tuning 하려면:

**메모리 요구량:**
- 모델 파라미터 (FP16): 7B * 2 bytes = 14 GB
- Gradient: 14 GB
- Optimizer 상태 (Adam): 파라미터당 8 bytes = 56 GB
- Activation: 배치 크기와 시퀀스 길이에 따라 가변
- **총합: 최소 80~100+ GB GPU 메모리**

**저장 문제:**
- 각 태스크/로봇마다 7B 모델의 완전한 복사본이 필요
- 10개 로봇에 적응하면 10 * 14 GB = 140 GB 저장 공간

**학습 시간:**
- 7B 파라미터 모두의 gradient 계산과 업데이트

이 비용이 PEFT(Parameter-Efficient Fine-Tuning) 연구를 촉발했다.

### 2. PEFT의 핵심 아이디어

사전학습된 모델 대부분을 **고정(freeze)**하고, **소수의 파라미터만** 학습하여 태스크에 적응한다.

**왜 이것이 가능한가?**
- 사전학습된 모델은 이미 풍부한 표현을 가지고 있다
- 파인튜닝에서의 가중치 변화량(delta W)은 실제로 **저차원(low-rank)**이다
- 전체 파라미터 공간 중 극히 일부만 변경해도 태스크 적응이 충분

### 3. PEFT 방법들의 분류

| 방법 | 접근 방식 | 대표 기법 |
|------|----------|----------|
| **Adapter** | 기존 레이어 사이에 작은 모듈 삽입 | Adapter Layers, Prefix Tuning |
| **Soft Prompt** | 입력에 학습 가능한 토큰 추가 | Prompt Tuning, P-Tuning |
| **재파라미터화** | 가중치 변화를 효율적으로 표현 | **LoRA**, IA3 |

이 중 LoRA가 현재 가장 널리 사용되며, OpenVLA도 LoRA를 사용한다.

### 4. LoRA (Low-Rank Adaptation) -- 핵심 기법

#### 기본 아이디어

파인튜닝 시 가중치 변화 delta_W가 **낮은 랭크(low-rank)**라는 가설에 기반한다.

원래 가중치가 W (d x d 행렬)이고, 파인튜닝 후 W + delta_W가 되어야 한다면:

- Full fine-tuning: delta_W를 직접 학습 (d * d개 파라미터)
- LoRA: delta_W = B * A로 분해 (B는 d x r, A는 r x d, r << d)

여기서 r은 **rank**로, 보통 4, 8, 16, 32 같은 작은 값을 사용한다.

#### 파라미터 절약량

예시: d = 4096 (Llama 2 7B의 hidden dimension)

| 방법 | 파라미터 수 |
|------|-----------|
| Full delta_W | 4096 * 4096 = 16.8M |
| LoRA (r=8) | 4096 * 8 + 8 * 4096 = 65.5K |
| LoRA (r=32) | 4096 * 32 + 32 * 4096 = 262K |

**r=8일 때 약 256배 절약, r=32일 때 약 64배 절약.**

#### LoRA의 학습 과정

1. 사전학습된 가중치 W를 **고정(freeze)**
2. 두 개의 작은 행렬 A와 B를 추가 (A는 랜덤 초기화, B는 0으로 초기화)
3. Forward pass: output = W * x + (B * A) * x * (alpha / r)
4. W는 업데이트하지 않고, A와 B만 학습
5. 학습 완료 후: W_new = W + B * A로 병합 가능

**B를 0으로 초기화하는 이유:** 학습 시작 시 delta_W = B * A = 0이 되어, 사전학습된 모델과 동일하게 시작. 학습이 안정적.

**alpha (스케일링 계수):** LoRA 출력의 크기를 조절. 보통 alpha = r 또는 alpha = 2r로 설정.

#### LoRA를 어디에 적용하는가

Transformer의 어떤 가중치 행렬에든 적용 가능하지만, 일반적으로:

- **Attention**: Q, K, V, O projection 행렬 (가장 일반적)
- **FFN**: Up, Gate, Down projection 행렬 (추가 적용 시)

OpenVLA는 **attention의 Q, K, V, O 모두에** LoRA를 적용한다.

#### 랭크 r의 선택

| r 값 | 학습 파라미터 | 표현력 | 사용 시나리오 |
|------|------------|--------|-------------|
| 4 | 매우 적음 | 제한적 | 매우 유사한 태스크로의 적응 |
| 8~16 | 적음 | 보통 | 일반적인 파인튜닝 |
| 32~64 | 보통 | 높음 | 도메인이 다른 태스크 |
| 128+ | 많음 | 매우 높음 | Full fine-tuning에 근접 |

OpenVLA는 r = 32를 사용한다. 로봇 행동 생성은 텍스트와 상당히 다른 도메인이므로 적당히 높은 rank가 필요.

### 5. QLoRA (Quantized LoRA)

LoRA를 더 극단적으로 효율화한 방법이다.

**핵심 아이디어:**
- 고정된 사전학습 가중치 W를 **4-bit로 양자화**하여 저장
- LoRA 행렬 A, B만 일반 정밀도(BF16)로 학습
- Forward pass 시 W를 BF16으로 역양자화하여 계산

**메모리 절약:**
- 기존: W를 FP16(16-bit)으로 저장 = 14 GB (7B 모델)
- QLoRA: W를 4-bit으로 저장 = 약 3.5 GB (7B 모델)
- LoRA 파라미터는 별도로 BF16으로 유지

**QLoRA의 기술적 구성:**
- **4-bit NormalFloat (NF4)**: 정규분포를 가정한 최적 4-bit 양자화
- **Double Quantization**: 양자화 상수도 다시 양자화하여 추가 메모리 절약
- **Paged Optimizers**: GPU 메모리 부족 시 CPU로 오프로드

**결과:** 65B 모델을 단일 48GB GPU에서 파인튜닝 가능. 7B 모델은 단일 24GB GPU에서도 가능.

### 6. 다른 PEFT 기법들

#### Adapter Layers

- Transformer 레이어 사이에 작은 병목(bottleneck) 네트워크를 삽입
- Down-projection -> 활성화 -> Up-projection
- 원본 가중치는 고정, adapter만 학습
- 단점: 추론 시 추가 레이어 때문에 지연 발생

#### Prefix Tuning

- 각 Transformer 레이어의 Key와 Value 앞에 **학습 가능한 가상 토큰**을 추가
- 이 가상 토큰의 임베딩만 학습
- 입력 시퀀스가 길어지는 효과 -> attention 계산 비용 증가

#### Prompt Tuning

- 입력 임베딩 앞에 학습 가능한 "soft prompt" 벡터를 추가
- 가장 간단하지만, 모델 크기가 작을 때 성능이 제한적

#### LoRA의 우위

LoRA가 가장 널리 사용되는 이유:
1. **추론 시 추가 비용 없음**: 학습 후 W + B*A로 병합하면 원래 모델과 같은 구조
2. **구현이 간단**: 기존 모델 구조를 변경하지 않음
3. **성능이 우수**: Full fine-tuning에 근접한 성능
4. **유연한 배포**: 원본 모델 하나에 여러 LoRA adapter를 교체 가능

### 7. OpenVLA에서의 LoRA 활용

**시나리오:** 사전학습된 OpenVLA를 새로운 로봇(예: 다른 그리퍼)에 적응시키기

**과정:**
1. OpenVLA 7.6B 파라미터를 모두 고정
2. Attention의 Q, K, V, O에 LoRA (r=32) 추가 -> 약 107M 학습 파라미터
3. 새 로봇의 데모 데이터 (수백~수천 개)로 LoRA만 학습
4. 결과: 로봇별 LoRA adapter 파일은 약 200~400MB

**다중 로봇 배포:**
- 기본 OpenVLA 모델: 1개 (약 15GB)
- 로봇 A용 LoRA: +400MB
- 로봇 B용 LoRA: +400MB
- 로봇 C용 LoRA: +400MB
- **총합: 약 16.2GB** (Full fine-tuning이었다면 약 60GB)

LoRA adapter를 교체하기만 하면 같은 기본 모델로 다양한 로봇에 대응 가능.

---

## 연습 주제 (코드 없이 생각해보기)

1. **Low-Rank 직관**: 4x4 행렬을 rank-1 행렬 (4x1) * (1x4)로 표현하면 파라미터가 16개에서 8개로 줄어든다. 하지만 rank-1 행렬은 어떤 제약이 있는가? rank가 높아지면 표현력이 어떻게 변하는가?

2. **메모리 계산**: Llama 2 7B를 full fine-tuning할 때와 LoRA (r=32)로 fine-tuning할 때의 optimizer 메모리 사용량을 비교하라. Adam optimizer는 파라미터당 8 bytes가 필요하다.

3. **랭크 선택**: OpenVLA가 r=32를 사용하는 이유를 추론하라. 만약 r=4를 사용하면 어떤 문제가 생길 수 있겠는가? r=256이면?

4. **QLoRA 트레이드오프**: 4-bit 양자화는 정밀도 손실을 유발한다. 이 손실이 LoRA 학습에 어떤 영향을 미칠 수 있는가? 왜 LoRA 행렬은 양자화하지 않는가?

5. **다중 로봇 배포 설계**: 5종류의 로봇에 OpenVLA를 배포한다. Full fine-tuning vs LoRA 방식의 총 저장 공간, GPU 메모리 요구량, 로봇 간 전환 시간을 비교하라.

6. **PEFT 방법 비교**: LoRA, Adapter, Prefix Tuning의 추론 시 비용을 비교하라. 실시간 로봇 제어에서 왜 LoRA가 유리한가?

---

## 다음 노트

[Modern Vision으로](../11-modern-vision/) -- 이제 LLM의 기반을 완료했다. 다음은 VLA의 또 다른 축인 vision 쪽으로, 최신 비전 모델들이 LLM과 어떻게 결합되는지 살펴본다.
