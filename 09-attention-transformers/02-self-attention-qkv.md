# Self-Attention과 Q/K/V (Self-Attention with Query, Key, Value)

## 왜 알아야 하는가 (Why This Matters for VLA)

Self-Attention은 **Transformer의 심장**이다. VLA 모델의 모든 구성 요소 -- Vision Encoder, Language Model, Action Decoder -- 가 Self-Attention 위에 세워져 있다. Q(Query), K(Key), V(Value)의 개념을 확실히 이해하면, 이후 Transformer부터 VLA까지 모든 것이 자연스럽게 이해된다.

VLA와의 연결 고리:
- VLA의 Llama 2 백본은 매 레이어마다 **Self-Attention**을 수행한다
  - 입력 토큰(이미지 패치 + 텍스트 토큰 + 액션 토큰)이 서로를 참조
  - "빨간 컵"이라는 텍스트 토큰이 이미지 속 빨간 컵 패치에 높은 attention weight를 부여
- Q/K/V는 단순한 수학 표기가 아니라, **역할(role)의 구분**이다
  - VLA에서 Query = "무엇을 찾고 있는가?", Key = "나는 무엇을 가지고 있는가?", Value = "내가 전달할 정보"
- Scaled Dot-Product Attention은 **VLA 추론(inference)의 70% 이상**의 연산을 차지한다
  - 이 수식을 이해해야 Flash Attention, KV-cache 등 최적화 기법을 이해할 수 있다

---

## 핵심 개념 (Core Concepts)

### 1. Cross-Attention에서 Self-Attention으로

이전 노트에서 배운 Bahdanau Attention은 **Cross-Attention**이다: 디코더(질문하는 쪽)와 인코더(참조되는 쪽)가 서로 다른 시퀀스이다.

```
Cross-Attention (이전 노트):
  디코더 상태(s) ──질문──→ 인코더 상태(h1, h2, h3)
  "나는 어디를 참조해야 하나?"    "여기 우리가 있어!"

  → 두 개의 다른 시퀀스 사이에서 작동
  → 예: 번역에서 출력 언어가 입력 언어를 참조

Self-Attention (이 노트):
  같은 시퀀스 내에서 각 위치가 다른 모든 위치를 참조

  "나는" ──→ "나는", "학생", "입니다"
  "학생" ──→ "나는", "학생", "입니다"    ← 자기 자신도 포함!
  "입니다" → "나는", "학생", "입니다"

  → 하나의 시퀀스가 자기 자신을 참조
  → 각 단어가 문맥 속에서 다른 단어들과의 관계를 파악
```

**Self-Attention이 필요한 이유**:
```
예문: "그 은행에서 돈을 인출했다" vs "강 은행에 앉아 있었다"

"은행"이라는 단어의 의미는 주변 단어에 의해 결정된다:
  - "돈", "인출" → 금융 기관
  - "강", "앉아" → 강둑(river bank)

Self-Attention은 "은행"이 문장의 다른 모든 단어를 참조하여
스스로의 의미를 결정할 수 있게 한다.

RNN은 이것을 순차적으로만 처리했다 (먼 단어일수록 정보 손실).
Self-Attention은 모든 위치를 동시에, 직접 참조한다.
```

### 2. Query, Key, Value의 직관적 이해

Self-Attention은 Q(Query), K(Key), V(Value)라는 세 가지 역할을 도입한다. 이것은 **도서관 검색 시스템**에 비유할 수 있다.

```
도서관 비유:

Query (질문):   "인공지능에 대한 책을 찾고 싶어요"
                → 내가 찾고 있는 것

Key (키워드):   각 책의 제목/키워드 태그
                → 각 항목이 자신을 설명하는 정보
                → "머신러닝 입문", "요리 레시피", "AI 윤리학"

Value (값):     각 책의 실제 내용
                → 매칭되었을 때 전달할 정보

검색 과정:
  1. Query("인공지능")와 모든 Key를 비교
  2. 가장 관련 높은 Key를 가진 책을 찾음
  3. 해당 책의 Value(내용)를 반환
```

**Self-Attention에서의 Q/K/V**:
```
입력: "나는 학생입니다" → 각 토큰이 벡터로 표현됨

각 토큰은 동시에 세 가지 역할을 수행한다:
  1. Query:  "나는 어떤 정보가 필요한가?" (질문자)
  2. Key:    "나는 어떤 정보를 가지고 있는가?" (설명자)
  3. Value:  "나의 실제 정보는 이것이다" (내용)

"학생"이 다른 단어를 참조할 때:
  Query = "학생"의 Q벡터 → "나는 누구의 학생인지, 무엇을 하는지 알고 싶다"
  Key   = 각 단어의 K벡터 → "나는", "학생", "입니다" 각각의 설명
  Value = 각 단어의 V벡터 → 각 단어가 실제로 전달할 정보

매칭: Q("학생") * K("나는") = 높은 score → "학생"은 "나는"에 주목
      → V("나는")의 정보를 많이 받아옴
```

### 3. Q/K/V의 생성: 선형 변환 (Linear Projection)

Q, K, V는 입력 벡터에 **서로 다른 가중치 행렬**을 곱하여 생성한다.

입력 토큰 벡터 $x$ (차원: $d_{\text{model}}$)에서:

$$Q = x \, W_Q \quad (W_Q: d_{\text{model}} \times d_k)$$

$$K = x \, W_K \quad (W_K: d_{\text{model}} \times d_k)$$

$$V = x \, W_V \quad (W_V: d_{\text{model}} \times d_v)$$

왜 같은 $x$에서 세 가지를 만드는가? 같은 사람이라도:
- 질문할 때의 관점 (Query): "나는 뭘 찾고 있지?"
- 자기를 설명할 때 (Key): "나는 이런 특성이 있어"
- 전달할 실제 정보 (Value): "나의 핵심 내용은 이것이야"

이 세 관점이 다르기 때문에, 서로 다른 가중치 행렬로 변환한다. $W_Q, W_K, W_V$는 학습을 통해 최적화된다.

**차원의 의미**:
```
VLA(Llama 2-7B) 기준:

d_model = 4096     (모델의 기본 차원)
d_k = d_v = 128    (Head당 차원, 나중에 Multi-Head에서 설명)

W_Q shape: (4096, 128)  → 4096차원 입력을 128차원 Query로 변환
W_K shape: (4096, 128)  → 4096차원 입력을 128차원 Key로 변환
W_V shape: (4096, 128)  → 4096차원 입력을 128차원 Value로 변환

직관: 4096차원의 풍부한 정보에서
      128차원의 특정 관점(역할)을 추출하는 것
```

### 4. Scaled Dot-Product Attention 수식

이것이 Transformer의 핵심 수식이다. VLA의 모든 어텐션은 이 공식을 사용한다.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

하나씩 분해하면:

**Step 1**: $QK^T$ (Score 계산)
→ 모든 Query와 모든 Key의 내적(dot product)
→ 결과: (시퀀스 길이 × 시퀀스 길이) 크기의 score 행렬

**Step 2**: $/ \sqrt{d_k}$ (스케일링)
→ score를 $\sqrt{d_k}$로 나눔 (이유는 아래에서 설명)

**Step 3**: $\text{softmax}(\cdots)$ (가중치 계산)
→ 각 행에 softmax 적용
→ 각 Query에 대해 모든 Key의 가중치 합이 1이 됨

**Step 4**: $\cdots \times V$ (가중 합산)
→ attention weight로 Value를 가중 합산
→ 최종 출력: 각 위치의 "문맥을 반영한 표현"

**구체적 숫자 예시**:
```
3개 토큰, d_k = 4인 간단한 예:

Q = [[1,0,1,0],     K = [[1,1,0,0],     V = [[1,0,0,1],
     [0,1,0,1],          [0,0,1,1],          [0,1,1,0],
     [1,1,0,0]]          [1,0,1,0]]          [1,1,0,0]]

Step 1: Q @ K^T
  = [[1*1+0*1+1*0+0*0,  1*0+0*0+1*1+0*1,  1*1+0*0+1*1+0*0],   [[1, 1, 2],
     [0*1+1*1+0*0+1*0,  0*0+1*0+0*1+1*1,  0*1+1*0+0*1+1*0],  =  [1, 1, 0],
     [1*1+1*1+0*0+0*0,  1*0+1*0+0*1+0*1,  1*1+1*0+0*1+0*0]]     [2, 0, 1]]

Step 2: / sqrt(4) = / 2
  = [[0.5, 0.5, 1.0],
     [0.5, 0.5, 0.0],
     [1.0, 0.0, 0.5]]

Step 3: softmax (각 행에 적용)
  ≈ [[0.26, 0.26, 0.48],     ← 토큰1은 토큰3에 가장 주목
     [0.39, 0.39, 0.22],     ← 토큰2는 토큰1,2에 비슷하게 주목
     [0.51, 0.19, 0.30]]     ← 토큰3은 토큰1에 가장 주목

Step 4: @ V → 최종 출력 (각 토큰의 문맥 반영 표현)
```

### 5. 왜 sqrt(d_k)로 나누는가 (Why Scale?)

이 스케일링은 단순해 보이지만, 없으면 학습이 **불안정**해진다.

**문제**: $d_k$가 클 때 dot product 값이 너무 커진다.

$Q$와 $K$의 각 원소가 평균 $0$, 분산 $1$인 독립 확률 변수라면:

$$\text{dot product} = q_1 k_1 + q_2 k_2 + \cdots + q_{d_k} k_{d_k}$$

각 항의 분산 $= 1 \times 1 = 1$, 합의 분산 $= d_k$ (독립이므로 분산이 더해짐)
→ dot product의 표준편차 $= \sqrt{d_k}$

```
구체적 예시:
  d_k = 64:   dot product의 표준편차 ≈ 8
  d_k = 128:  dot product의 표준편차 ≈ 11.3
  d_k = 512:  dot product의 표준편차 ≈ 22.6

  score가 [-20, +20] 범위로 퍼지면...
```

```
Softmax의 문제:

softmax([20, -20, 0]) ≈ [1.00, 0.00, 0.00]  ← 거의 one-hot!
softmax([2, -2, 0])   ≈ [0.71, 0.01, 0.10]  ← 적절한 분포

score 값이 너무 크면:
  1. Softmax가 극단적인 분포(거의 one-hot)를 만든다
  2. Gradient가 거의 0이 된다 (softmax의 포화 영역)
  3. 학습이 멈추거나 불안정해진다

해결: sqrt(d_k)로 나누어 score의 분산을 1로 정규화
  → Softmax가 적절한 분포를 유지
  → Gradient가 잘 흐름
  → 안정적인 학습
```

```
VLA에서의 실제 값:
  Llama 2-7B: d_k = 128 → sqrt(128) ≈ 11.3으로 나눔
  ViT-Large:  d_k = 64  → sqrt(64) = 8로 나눔
```

### 6. Self-Attention의 출력: 문맥을 반영한 표현

Self-Attention의 출력은 **문맥 정보가 반영된 새로운 표현**이다.

```
입력 vs 출력:

입력: 각 토큰의 독립적인 표현
  "bank" → [0.3, 0.7, 0.1, ...]   (고정된 의미, 문맥 무관)

Self-Attention 후:
  "river bank" → [0.1, 0.9, 0.3, ...]  (강둑의 의미로 업데이트)
  "bank account" → [0.8, 0.2, 0.6, ...] (금융기관의 의미로 업데이트)

→ 같은 단어라도 문맥에 따라 다른 표현으로 변환된다!
→ 이것이 Self-Attention의 핵심 능력
```

**VLA에서의 예시**:
```
VLA 입력 시퀀스: [이미지 패치들] + [텍스트 토큰들]

Self-Attention 전:
  이미지 패치 137번: "빨간색 물체가 있는 영역" (시각 정보만)
  텍스트 토큰 "컵":   "컵이라는 단어" (언어 정보만)

Self-Attention 후:
  이미지 패치 137번: "텍스트에서 언급된 빨간 컵이 있는 영역" (시각+언어)
  텍스트 토큰 "컵":   "이미지 속 137번 패치에 보이는 빨간 컵" (언어+시각)

→ 서로 다른 모달리티(시각, 언어)의 정보가 Self-Attention을 통해 융합된다
→ 이것이 VLA가 "보고 이해하고 행동하는" 핵심 원리
```

### 7. Self-Attention vs RNN: 근본적 차이

**RNN**: 토큰1 → 토큰2 → 토큰3 → ... → 토큰N (순차적, $O(N)$ 단계)

문제:
- 토큰1과 토큰N 사이: $N-1$ 단계를 거쳐야 정보 전달
- 먼 거리의 정보가 점점 희미해짐 (장기 의존성 문제)
- 순차 처리 → GPU 병렬화 불가능 → 느림

**Self-Attention**: 토큰1 ↔ 토큰2 ↔ 토큰3 ↔ ... ↔ 토큰N (모든 쌍이 직접 연결, $O(1)$ 단계)

장점:
- 토큰1과 토큰N 사이: 단 1단계로 직접 참조
- 먼 거리도 가까운 거리도 동일하게 처리
- 모든 쌍을 동시에 계산 → GPU 병렬화 가능 → 빠름

단점:
- 연산량: $O(N^2 \cdot d)$ -- 시퀀스 길이의 제곱!
- VLA에서 이미지 패치(256개) + 텍스트(수백 토큰) → 수십만 개의 쌍
- → Flash Attention 등 최적화가 필수적인 이유

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **Q/K/V 직관 연습**: "나는 맛있는 사과를 먹었다"에서 "사과"가 Query일 때, 다른 단어들의 Key와의 유사도(attention weight)가 어떻게 분포될지 예상해보라. "맛있는"과 "먹었다" 중 어디에 더 높은 가중치가 갈까?

2. **스케일링 계산**: $d_k = 256$일 때 $\sqrt{d_k}$는 얼마인가? 스케일링 없이 score 값이 평균적으로 어떤 범위에 분포하는지, 스케일링 후에는 어떤 범위가 되는지 계산해보라.

3. **연산량 비교**: 시퀀스 길이 $N=1000$인 Self-Attention의 score 행렬 크기는? $N=4000$이면? 시퀀스 길이가 4배 늘면 연산량은 몇 배 증가하는가? VLA가 긴 시퀀스를 다루는 것이 왜 어려운지 연결지어 생각해보라.

4. **Cross-Attention vs Self-Attention 구분**: VLA에서 "이미지 패치들이 서로를 참조"하는 것은 Self-Attention인가 Cross-Attention인가? "텍스트 토큰이 이미지 패치를 참조"하는 것은? 각각의 Q, K, V가 어디에서 오는지 명확히 구분해보라.

5. **문맥 표현의 변화**: "I went to the bank to fish"와 "I went to the bank to deposit"에서 "bank"의 Self-Attention 출력이 어떻게 달라질지 직관적으로 설명해보라. 어떤 단어에 높은 attention weight가 갈까?

---

## 다음 노트 (Next Note)

Self-Attention의 원리(Q/K/V, Scaled Dot-Product)를 이해했다. 이제 이 Self-Attention을 **완전한 모델**로 조립할 차례이다. 인코더-디코더를 쌓고, 잔차 연결과 Layer Norm을 더하면 -- 바로 그 유명한 **Transformer**가 된다.

**다음**: [Transformer 아키텍처](./03-transformer-architecture.md) - "Attention Is All You Need" 논문의 전체 구조. 인코더/디코더 스택, Masked Self-Attention, Cross-Attention, 잔차 연결, Layer Norm까지.
