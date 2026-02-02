# Transformer 구현 (Transformer Implementation)

## 왜 알아야 하는가 (Why This Matters for VLA)

지금까지 Transformer의 개념을 배웠다. 이 노트에서는 이론을 **PyTorch 코드 수준**으로 구체화한다. 실제 구현에서 발생하는 핵심 문제들 -- 마스킹, 효율적 추론, 메모리 최적화 -- 을 이해하면 VLA 모델을 직접 다룰 수 있게 된다.

VLA와의 연결 고리:
- VLA를 Fine-tuning하려면 **Transformer의 PyTorch 구현 구조**를 읽을 수 있어야 한다
  - HuggingFace의 LlamaForCausalLM 코드를 이해해야 VLA를 수정할 수 있다
- **마스킹(Masking)**은 VLA 학습에서 결정적으로 중요하다:
  - Causal Mask: 미래 토큰을 보지 못하게 하는 자기회귀 마스크
  - Padding Mask: 배치 내 길이가 다른 시퀀스 처리
  - VLA는 이미지/텍스트/액션 구간별로 다른 마스킹 전략을 사용할 수 있다
- **KV-cache**는 VLA 추론(inference) 속도의 핵심이다:
  - 로봇은 실시간으로 동작해야 한다 (지연 = 실패)
  - KV-cache가 없으면 토큰 하나 생성할 때마다 전체를 다시 계산
  - KV-cache 원리를 알아야 추론 최적화를 이해할 수 있다

---

## 핵심 개념 (Core Concepts)

### 1. 전체 아키텍처를 PyTorch 모듈로 매핑

Transformer의 각 구성 요소가 PyTorch에서 어떤 모듈에 대응하는지 정리하자.

```
Transformer 전체 구조 → PyTorch 매핑:

class TransformerModel:
  ├── Embedding         (nn.Embedding)     : 토큰 → 벡터
  ├── PositionalEncoding (직접 구현 또는 nn.Embedding)
  ├── TransformerBlock × N:
  │   ├── RMSNorm       (직접 구현)         : 정규화
  │   ├── MultiHeadAttention:
  │   │   ├── W_Q       (nn.Linear)        : Query 변환
  │   │   ├── W_K       (nn.Linear)        : Key 변환
  │   │   ├── W_V       (nn.Linear)        : Value 변환
  │   │   ├── Attention  (행렬곱 + softmax) : Scaled Dot-Product
  │   │   └── W_O       (nn.Linear)        : 출력 변환
  │   ├── RMSNorm       (직접 구현)
  │   └── SwiGLU_FFN:
  │       ├── W_gate    (nn.Linear)        : 게이트 변환
  │       ├── W_up      (nn.Linear)        : 확장 변환
  │       └── W_down    (nn.Linear)        : 축소 변환
  └── OutputHead       (nn.Linear)         : 벡터 → 토큰 확률
```

**Llama 2-7B의 구체적 설정**:
```
모델 설정 (Config):
  vocab_size:     32000    (토큰 사전 크기)
  d_model:        4096     (모델 차원 = hidden_size)
  n_layers:       32       (Transformer Block 수)
  n_heads:        32       (Attention Head 수)
  d_ff:           11008    (FFN 중간 차원 = intermediate_size)
  max_seq_len:    4096     (최대 시퀀스 길이)
  norm_eps:       1e-6     (RMSNorm의 epsilon)

총 파라미터 계산:
  Embedding:      32000 × 4096 = 131,072,000        (1.3억)
  Attention × 32: 32 × (4 × 4096 × 4096) = 2,147,483,648  (21.5억)
  FFN × 32:       32 × (3 × 4096 × 11008) = 4,328,521,728  (43.3억)
  Output Head:    4096 × 32000 = 131,072,000         (1.3억)
  Norm 등:       약간
  총합:           약 67억 → "7B" 모델
```

### 2. 마스킹 (Masking): 정보 흐름 제어

마스킹은 Transformer에서 **어떤 위치의 정보를 차단**하는 메커니즘이다. 두 가지 주요 마스크가 있다.

```
마스크의 적용 위치:

Attention(Q, K, V) = softmax((Q @ K^T / sqrt(d_k)) + Mask) @ V
                                                     ^^^^^^
                                                     여기에 더함!

Mask 행렬의 값:
  볼 수 있는 위치: 0       → score에 0을 더함 → 변화 없음
  볼 수 없는 위치: -inf    → score가 -inf → softmax 후 0
```

**Padding Mask (패딩 마스크)**:
```
배치(batch) 처리 시, 시퀀스 길이가 다른 경우:

시퀀스 1: ["I", "am", "happy", <PAD>, <PAD>]     길이 3
시퀀스 2: ["The", "cat", "is", "black", <PAD>]    길이 4
시퀀스 3: ["Hello", "world", <PAD>, <PAD>, <PAD>]  길이 2

<PAD> 토큰은 의미가 없으므로 attention에서 제외해야 한다.

Padding Mask (시퀀스 1의 경우):
           I    am   happy PAD  PAD
  I       [0    0    0    -inf -inf]
  am      [0    0    0    -inf -inf]
  happy   [0    0    0    -inf -inf]
  PAD     [-inf -inf -inf -inf -inf]
  PAD     [-inf -inf -inf -inf -inf]

→ PAD 위치의 Key에 대해 모든 Query가 -inf → attention weight 0
→ PAD 위치의 Query도 어디에도 주목하지 않음
```

**Causal Mask (인과 마스크)**:
```
자기회귀(autoregressive) 생성에서 미래 토큰을 차단:

Causal Mask (5 토큰):
           pos0  pos1  pos2  pos3  pos4
  pos0    [0    -inf  -inf  -inf  -inf]    ← 자기만 볼 수 있음
  pos1    [0     0    -inf  -inf  -inf]    ← pos0, pos1만
  pos2    [0     0     0    -inf  -inf]    ← pos0~2만
  pos3    [0     0     0     0    -inf]    ← pos0~3만
  pos4    [0     0     0     0     0  ]    ← 전부 볼 수 있음

→ 하삼각 행렬 (Lower Triangular Matrix)
→ 각 위치는 자기 이전(과거)만 참조 가능

GPT/Llama/VLA의 디코더에서 항상 사용!
```

**VLA에서의 결합 마스크**:
```
VLA 입력 구조: [이미지 패치] [텍스트] [액션]

마스킹 전략 (모델에 따라 다름):

방법 1: 순수 Causal Mask
  [이미지 패치] → 이전 패치만 참조
  [텍스트]      → 이미지 + 이전 텍스트만 참조
  [액션]        → 이미지 + 텍스트 + 이전 액션만 참조

방법 2: 이미지 내부는 양방향, 나머지는 Causal
  [이미지 패치] → 모든 이미지 패치를 양방향 참조 (ViT처럼)
  [텍스트]      → 이미지 전체 + 이전 텍스트만 참조
  [액션]        → 이미지 전체 + 텍스트 전체 + 이전 액션만 참조

  이 경우 마스크:
              이미지     텍스트     액션
  이미지    [양방향    -inf       -inf    ]
  텍스트    [볼 수 있음 Causal    -inf    ]
  액션      [볼 수 있음 볼 수 있음 Causal ]

→ 마스크 설계가 VLA의 성능에 직접 영향!
```

### 3. 학습 vs 추론: 근본적 차이

Transformer의 학습과 추론은 **근본적으로 다른 방식**으로 작동한다.

```
학습 (Training): 병렬 처리

입력: "I am a student" (전체가 한 번에 입력)

       I    am    a    student
  Q:  [q1]  [q2]  [q3]  [q4]     ← 4개 토큰의 Q를 한 번에 계산
  K:  [k1]  [k2]  [k3]  [k4]     ← 4개 토큰의 K를 한 번에 계산
  V:  [v1]  [v2]  [v3]  [v4]     ← 4개 토큰의 V를 한 번에 계산

  Score = Q @ K^T  (4×4 행렬 연산 한 번!)
  + Causal Mask
  → softmax → @ V
  → 4개 출력을 한 번에 계산

  타겟: "am a student <EOS>"
  → 모든 위치의 손실을 한 번에 계산
  → 매우 효율적 (GPU 병렬화)

추론 (Inference): 순차 생성

Step 1: 입력 "I"
  → 다음 토큰 예측: "am"

Step 2: 입력 "I am"
  → 다음 토큰 예측: "a"

Step 3: 입력 "I am a"
  → 다음 토큰 예측: "student"

  매 Step마다 전체 시퀀스를 다시 계산해야 하는가?
  → 이것이 KV-cache가 해결하는 문제!
```

### 4. KV-cache: 추론 속도의 핵심

KV-cache는 이전에 계산한 Key와 Value를 **재사용**하여 추론 속도를 획기적으로 높인다.

```
KV-cache 없이 (Naive 추론):

Step 1: "I"                → K:[k1], V:[v1] → 출력 "am"
Step 2: "I", "am"          → K:[k1,k2], V:[v1,v2] → 출력 "a"
Step 3: "I", "am", "a"     → K:[k1,k2,k3], V:[v1,v2,v3] → 출력 "student"

문제: Step 3에서 k1, v1을 또 계산! (이미 Step 1에서 했는데!)
  → 이전 토큰의 K, V를 매번 다시 계산 → 엄청난 낭비

KV-cache 사용:

Step 1: "I"  → k1, v1 계산 → cache에 저장 → 출력 "am"
  cache: K=[k1], V=[v1]

Step 2: "am" → k2, v2만 새로 계산 → cache에 추가 → 출력 "a"
  cache: K=[k1,k2], V=[v1,v2]
  (k1, v1은 cache에서 가져옴 -- 다시 계산하지 않음!)

Step 3: "a"  → k3, v3만 새로 계산 → cache에 추가 → 출력 "student"
  cache: K=[k1,k2,k3], V=[v1,v2,v3]

핵심: 새 토큰의 K, V만 계산하고, 이전 것은 cache에서 재사용!
```

**KV-cache의 효율성**:
```
N개 토큰을 생성할 때:

KV-cache 없이:
  총 연산: 1 + 2 + 3 + ... + N = N(N+1)/2 ≈ N^2/2
  (매 Step마다 전체 시퀀스 처리)

KV-cache 사용:
  총 연산: N + (Attention에서 cache 참조)
  (매 Step마다 새 토큰 1개만 처리)

  N=1000일 때:
    없이: 약 500,000 단위 연산
    사용: 약 1,000 단위 연산 + cache 참조
    → 약 500배 빠름!
```

**KV-cache의 메모리 비용**:
```
Llama 2-7B에서 KV-cache의 메모리:

한 레이어당:
  K cache: (batch_size, n_heads, seq_len, d_k) = (1, 32, seq_len, 128)
  V cache: (batch_size, n_heads, seq_len, d_k) = (1, 32, seq_len, 128)

  한 레이어, 한 토큰당 K+V: 2 × 32 × 128 = 8192 (float16: 16KB)

32레이어, 시퀀스 길이 2048:
  32 × 2048 × 16KB = 1,048,576 KB ≈ 1GB

→ 시퀀스가 길어질수록 KV-cache 메모리가 선형 증가
→ VLA에서 이미지 패치가 많으면 KV-cache가 매우 커짐
→ Grouped Query Attention (GQA)로 K/V Head를 줄여 메모리 절약
```

### 5. Prefill과 Decode 단계 (Two-Phase Inference)

실제 추론은 두 단계로 나뉜다.

```
Phase 1: Prefill (프리필)
  - 전체 입력 프롬프트를 한 번에 처리
  - 모든 토큰의 K, V를 계산하여 cache에 저장
  - 병렬 처리 가능 → GPU를 최대한 활용

  VLA에서: 이미지 패치 + 텍스트 명령을 한 번에 처리
  [패치1][패치2]...[패치256]["빨간"]["컵을"]["집어라"]
  → 모든 K, V를 cache에 저장

Phase 2: Decode (디코드)
  - 토큰을 하나씩 순차 생성
  - 새 토큰의 Q만 계산, K/V는 cache에서 참조
  - 순차 처리 → GPU 활용도가 낮음 (memory-bound)

  VLA에서: 액션 토큰을 하나씩 생성
  [액션1] 생성 → [액션2] 생성 → ... → [액션7] 생성

병목:
  Prefill: 한 번만 수행, compute-bound (연산량이 병목)
  Decode:  여러 번 수행, memory-bound (메모리 대역폭이 병목)
  → VLA 실시간 제어에서 Decode 속도가 핵심
```

### 6. 주요 구현 세부사항 정리

실제 코드에서 마주치는 핵심 구현 패턴들이다.

```
1. Attention Score 계산의 실제 구현:

  # 효율적인 구현 (BatchedMatMul)
  # Q: (batch, n_heads, seq_len, d_k)
  # K: (batch, n_heads, seq_len, d_k)
  # V: (batch, n_heads, seq_len, d_k)

  scores = Q @ K.transpose(-2, -1) / sqrt(d_k)   # (batch, n_heads, seq, seq)
  scores = scores + mask                            # 마스크 적용 (0 또는 -inf)
  weights = softmax(scores, dim=-1)                 # 각 행에 softmax
  output = weights @ V                              # 가중 합산

2. RoPE 적용 위치:
  Q와 K에만 적용 (V에는 적용하지 않음!)
  → Q = apply_rope(x @ W_Q, positions)
  → K = apply_rope(x @ W_K, positions)
  → V = x @ W_V  (RoPE 없이)

3. RMSNorm 구현:
  output = x * (1 / sqrt(mean(x^2) + eps)) * gamma
  → Layer Norm과 달리 평균을 빼지 않음
  → 더 빠르고 비슷한 성능

4. SwiGLU 구현:
  gate = SiLU(x @ W_gate)    # 게이트 활성화
  up = x @ W_up              # 확장
  output = (gate * up) @ W_down  # 필터링 후 축소
```

```
5. 학습 시 주의사항:

  Label Shifting:
    입력: [BOS, I, am, a, student]
    타겟: [I, am, a, student, EOS]
    → 입력을 한 칸 오른쪽으로 shift하여 다음 토큰 예측

  Loss Masking:
    이미지 패치와 텍스트 입력 부분은 loss를 계산하지 않는 경우도 있음
    VLA에서는 보통 액션 토큰에 대해서만 loss를 계산
    → loss mask: [0,0,...,0, 1,1,...,1]
                  (이미지/텍스트) (액션)

6. 추론 시 생성 전략:
  Greedy:    가장 확률 높은 토큰 선택 (결정적)
  Sampling:  확률 분포에서 샘플링 (확률적)
  Top-k:     상위 k개 토큰에서만 샘플링
  Top-p:     누적 확률 p까지의 토큰에서만 샘플링

  VLA에서: 로봇 동작은 보통 Greedy 또는 낮은 temperature sampling
  → 로봇은 안정적이고 예측 가능한 동작이 중요
```

### 7. 전체 추론 파이프라인 (VLA 관점)

```
VLA 추론 파이프라인 전체 흐름:

1. 이미지 전처리:
   카메라 이미지 (224×224×3)
   → ViT 패치 분할 (16×16 패치 → 14×14 = 196개 패치)
   → Vision Encoder (ViT) → 이미지 토큰 (196개, 각 1024차원)
   → Projection (1024 → 4096) → Llama 입력 차원에 맞춤

2. 텍스트 토크나이즈:
   "빨간 컵을 집어라" → Tokenizer → [토큰 ID 시퀀스]
   → Embedding → 텍스트 토큰 (각 4096차원)

3. Prefill:
   [이미지 토큰 | 텍스트 토큰] → Llama 2 (32 Transformer Blocks)
   → 모든 K, V를 cache에 저장

4. Decode (액션 생성):
   Step 1: 이전 출력 + cache → 다음 액션 토큰 예측 → [Δx]
   Step 2: cache 업데이트 → 다음 액션 토큰 예측 → [Δy]
   ...
   Step 7: → [gripper] (그리퍼 열기/닫기)

5. 액션 실행:
   [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
   → De-tokenize (토큰 → 연속값)
   → 로봇 제어기에 전달
   → 실제 로봇 동작 실행

6. 다음 시점:
   새 카메라 이미지 → 1번부터 반복 (실시간 제어 루프)
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **마스크 설계**: VLA 입력이 [이미지 패치 4개 | 텍스트 3개 | 액션 3개]일 때, 순수 Causal Mask (10x10 행렬)을 직접 그려보라. 그 다음, "이미지는 양방향, 나머지는 Causal"인 마스크를 그려보고 차이를 비교하라.

2. **KV-cache 메모리 계산**: d_model=2048, n_heads=16, d_k=128인 모델에서 24개 레이어를 사용한다. 시퀀스 길이 1024일 때 KV-cache의 float16 메모리 사용량을 계산하라.

3. **Prefill vs Decode 시간**: 입력 프롬프트가 500 토큰이고, 100 토큰을 생성해야 한다. Prefill은 500 토큰을 병렬 처리하고, Decode는 100 토큰을 순차 생성한다. 어느 단계가 더 시간이 많이 걸릴지 직관적으로 추론해보라. (힌트: 병렬 vs 순차)

4. **Loss Masking의 중요성**: VLA 학습에서 이미지 패치와 텍스트에 대해서는 loss를 계산하지 않고, 액션 토큰에 대해서만 loss를 계산하는 이유를 생각해보라. 만약 모든 토큰에 loss를 걸면 어떤 문제가 생길까?

5. **실시간 제어의 제약**: VLA 로봇이 10Hz(초당 10회)로 동작해야 한다면, 한 번의 추론에 최대 몇 ms를 쓸 수 있는가? Llama 2-7B의 전체 추론 시간이 이 제약을 만족하려면 어떤 최적화가 필요할지 생각해보라.

---

## 다음 노트 (Next Note)

Transformer의 이론부터 구현까지 모든 것을 다뤘다. 이 구조를 **거대한 규모로 학습**하면 놀라운 능력이 나타난다. 다음은 Transformer를 기반으로 한 **Large Language Model(LLM)**의 세계이다. GPT에서 Llama 2까지, VLA의 "두뇌"를 구성하는 LLM을 알아보자.

**다음**: [../10-llms/](../10-llms/) - 대규모 언어 모델(LLM)의 세계. Pre-training, Scaling Law, In-context Learning, 그리고 VLA의 백본인 Llama 2의 구조까지.
