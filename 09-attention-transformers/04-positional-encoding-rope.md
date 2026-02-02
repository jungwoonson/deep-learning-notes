# 위치 인코딩 (Positional Encoding)

## 왜 알아야 하는가 (Why This Matters for VLA)

Self-Attention은 강력하지만, 치명적인 약점이 있다: **토큰의 순서를 모른다**. "고양이가 개를 쫓는다"와 "개가 고양이를 쫓는다"를 동일하게 처리한다. 위치 인코딩(Positional Encoding)은 이 문제를 해결하여 Transformer에 **순서 감각**을 부여한다.

VLA와의 연결 고리:
- VLA에서 순서는 **생존적으로 중요**하다:
  - 로봇 동작 시퀀스 [위로 이동 → 그리퍼 열기 → 아래로 이동 → 그리퍼 닫기]
  - 순서가 바뀌면 물체를 집는 대신 놓쳐버린다
  - 이미지 패치의 공간적 위치도 순서 정보로 인코딩된다
- VLA의 핵심 모델인 **Llama 2는 RoPE(Rotary Position Embedding)**를 사용한다
  - RoPE는 현재 가장 널리 사용되는 위치 인코딩 방법
  - 긴 시퀀스(이미지 패치 수백 개 + 텍스트 + 액션)를 효과적으로 처리
  - RoPE를 이해하면 Llama 2의 내부가 보인다
- ViT(Vision Transformer)도 위치 인코딩으로 **패치의 공간 위치**를 표현한다
  - 위치 인코딩 없이는 이미지 왼쪽 위와 오른쪽 아래를 구분할 수 없다

---

## 핵심 개념 (Core Concepts)

### 1. 왜 위치 인코딩이 필요한가: 순열 동변성 (Permutation Equivariance)

Self-Attention의 수학적 성질을 살펴보면, 순서를 무시하는 이유가 명확해진다.

**Self-Attention의 핵심 성질: 순열 동변성(Permutation Equivariance)**

입력 토큰을 어떤 순서로 넣어도, 출력도 같은 순서로 바뀐다. 즉, Attention은 토큰의 "내용"만 보고, "위치"는 보지 않는다.

```
예시:
  입력 A: ["나는", "학생", "이다"]  → Self-Attention → [out1, out2, out3]
  입력 B: ["이다", "나는", "학생"]  → Self-Attention → [out3, out1, out2]

  출력 값 자체는 동일! 순서만 바뀜.
  → "나는 학생이다" = "이다 나는 학생" ← Self-Attention이 볼 때!
```

왜 그런가? Attention 수식에서 "위치"가 어디에도 등장하지 않는다:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

각 토큰의 Q, K, V는 오직 그 토큰의 내용(embedding)에서만 생성된다. 1번 위치든 100번 위치든, 같은 단어면 같은 Q/K/V를 생성.

```
대비: RNN은 순서를 자연스럽게 처리한다

RNN:  "나는" → h1 → "학생" → h2 → "이다" → h3
      순차 처리 → 위치 정보가 hidden state에 자연스럽게 포함

Self-Attention은 모든 토큰을 동시에 처리 → 순서 정보 상실
→ 별도로 위치 정보를 주입해야 한다!
```

### 2. 위치 인코딩의 기본 아이디어

각 위치에 고유한 **위치 벡터**를 만들어, 토큰 임베딩에 더한다.

```
기본 아이디어:

토큰 임베딩:     E("나는") = [0.3, 0.7, -0.1, 0.5, ...]
위치 인코딩:     PE(pos=0) = [0.0, 1.0,  0.0, 0.8, ...]
                 PE(pos=1) = [0.8, 0.6, -0.2, 0.3, ...]
                 PE(pos=2) = [0.9, -0.4, 0.9, -0.1, ...]

최종 입력 = 토큰 임베딩 + 위치 인코딩

pos=0: E("나는") + PE(0) = [0.3, 1.7, -0.1, 1.3, ...]
pos=1: E("학생") + PE(1) = [E + PE 결과]
pos=2: E("이다") + PE(2) = [E + PE 결과]

→ 같은 단어라도 위치가 다르면 다른 벡터가 됨
→ Self-Attention이 위치 차이를 인식할 수 있게 됨
```

위치 인코딩 방법은 크게 세 가지로 발전해왔다:
```
발전 순서:

1. 사인/코사인 (Sinusoidal)  -- 원래 Transformer (2017)
2. 학습 가능 (Learned)        -- BERT, GPT-2, ViT
3. RoPE (Rotary)              -- Llama 2, VLA (2023~)

각각의 특성과 장단점을 알아보자.
```

### 3. 사인/코사인 위치 인코딩 (Sinusoidal Positional Encoding)

원래 Transformer 논문에서 제안한 방법이다.

$$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

- $\text{pos}$: 토큰의 위치 (0, 1, 2, ...)
- $i$: 차원 인덱스 (0, 1, 2, ..., $d_{\text{model}}/2 - 1$)
- $d_{\text{model}}$: 모델 차원

```
직관적 이해: "이진 시계" 비유

이진법 시계:           사인/코사인 위치 인코딩:
pos=0: 0 0 0 0          저주파 ~~~~~~  중주파 ~~~  고주파 ~
pos=1: 0 0 0 1          sin, cos가 서로 다른 주파수로 진동
pos=2: 0 0 1 0          → 각 위치마다 고유한 패턴 생성
pos=3: 0 0 1 1
pos=4: 0 1 0 0          이진법처럼 낮은 비트는 빠르게 변하고
...                     높은 비트는 느리게 변한다

→ 서로 다른 주파수의 사인/코사인 조합으로
  각 위치에 고유한 "지문"을 부여
```

**사인/코사인의 핵심 장점**:

$PE(\text{pos}+k)$는 $PE(\text{pos})$의 선형 변환으로 표현할 수 있다:

$$PE(\text{pos}+k) = M_k \cdot PE(\text{pos}) \quad (M_k\text{는 위치 차이 } k\text{에만 의존하는 행렬})$$

의미:
- 모델이 "$k$칸 뒤에 있는 토큰"이라는 상대적 관계를 학습할 수 있다
- 절대 위치보다 상대 위치가 언어에서 더 중요하다 ("주어 다음에 동사"는 위치 3보다 "주어 뒤"가 핵심)

또 다른 장점: 학습 시 보지 못한 긴 시퀀스에도 적용 가능 (사인/코사인이므로 외삽 가능)

### 4. 학습 가능한 위치 인코딩 (Learned Positional Encoding)

BERT, GPT-2, ViT에서 사용하는 방법이다.

```
학습 가능한 위치 인코딩:

위치 임베딩 테이블을 만들고, 학습으로 최적화한다.

PE = Embedding(max_positions, d_model)

  PE[0] = [학습된 벡터]  → 위치 0에 해당하는 d_model 차원 벡터
  PE[1] = [학습된 벡터]  → 위치 1에 해당하는 벡터
  ...
  PE[511] = [학습된 벡터] → 최대 위치에 해당하는 벡터

장점:
  - 데이터에서 최적의 위치 표현을 자동으로 학습
  - 구현이 간단

단점:
  - 최대 길이가 고정됨 (학습 시 본 최대 길이까지만)
  - 학습에 본 적 없는 위치에는 일반화 불가능
  - 파라미터 수 증가: max_positions × d_model

ViT에서의 사용:
  - 이미지를 14×14 = 196개 패치로 나눈 후
  - 각 패치에 학습된 위치 임베딩을 더함
  - 2D 공간 위치를 1D 인덱스로 변환하여 사용
```

```
ViT의 위치 인코딩 시각화:

이미지를 패치로 나눔:
  [P0 ] [P1 ] [P2 ] [P3 ]
  [P4 ] [P5 ] [P6 ] [P7 ]
  [P8 ] [P9 ] [P10] [P11]
  [P12] [P13] [P14] [P15]

1D 순서: P0, P1, P2, ..., P15
위치 인코딩: PE[0], PE[1], PE[2], ..., PE[15]

학습 후 PE를 시각화하면:
  → 공간적으로 가까운 패치의 PE가 유사
  → 2D 구조를 1D 인코딩으로도 잘 학습!
```

### 5. RoPE: Rotary Position Embedding (회전 위치 임베딩)

RoPE는 Llama 2와 VLA에서 사용하는 **현재 가장 인기 있는** 위치 인코딩이다.

**RoPE의 핵심 아이디어**:

- 사인/코사인: 위치 정보를 벡터에 "더한다" (additive)
- RoPE: 위치 정보를 벡터를 "회전시켜" 반영한다 (rotary)

2차원 공간에서의 직관 — 벡터 $[x, y]$를 각도 $\theta$만큼 회전:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

RoPE는 이 회전을 고차원으로 확장한다:
- $d_{\text{model}}$ 차원 벡터를 2차원씩 쌍으로 묶어서 각 쌍을 서로 다른 각도로 회전시킨다
- 위치 $\text{pos}$인 토큰의 Query/Key: $q$와 $k$를 2차원 쌍으로 나눔 → 각 쌍을 $\text{pos} \cdot \theta_i$만큼 회전
- $\theta_i$는 차원마다 다른 기본 각도 (사인/코사인처럼 주파수가 다름)

**RoPE의 핵심 성질: 상대 위치의 자연스러운 인코딩**:

위치 $m$인 토큰의 Query: $q_m = R(m\theta) \cdot q$

위치 $n$인 토큰의 Key: $k_n = R(n\theta) \cdot k$

두 토큰의 attention score (dot product):

$$q_m^T k_n = (R(m\theta) \cdot q)^T (R(n\theta) \cdot k)$$

이 값은 놀랍게도 $m$과 $n$의 절대 위치가 아닌 $(m - n)$, 즉 **상대 위치에만 의존**한다!

수학적 성질:

$$R(m\theta)^T \cdot R(n\theta) = R((n-m)\theta)$$

의미: 토큰이 위치 5에 있든 위치 100에 있든, "$3$칸 떨어진 토큰"에 대한 attention은 동일. 상대적 거리만이 중요.

**RoPE가 VLA에서 중요한 이유**:
```
1. 긴 시퀀스 처리:
   VLA 입력 = 이미지 패치(256~576개) + 텍스트(수십~수백) + 액션(수십)
   → 총 수백~수천 토큰
   → RoPE는 학습 시 보지 못한 더 긴 시퀀스에도 비교적 잘 일반화

2. 상대 위치의 중요성:
   "컵이 테이블 위에 있다" → "위에"와 "컵"의 상대 위치가 중요
   로봇 동작에서도 "이전 동작 대비 변화"가 중요
   → RoPE가 상대 위치를 자연스럽게 인코딩

3. 효율성:
   학습 가능한 임베딩 테이블 불필요 → 파라미터 절약
   Q와 K에만 적용 (V에는 적용하지 않음) → 연산 절약
   회전 행렬은 희소 행렬 → 실제 연산량이 적음

4. 실제 사용 모델:
   Llama 2, Llama 3, Mistral, Qwen → 모두 RoPE 사용
   VLA가 이들을 백본으로 사용 → RoPE가 VLA의 표준
```

### 6. 세 가지 방법 비교 정리

```
                    Sinusoidal      Learned         RoPE
제안 연도            2017            2018~2019       2021
학습 여부            고정            학습 가능        고정 (연산)
위치 유형            절대 위치        절대 위치        상대 위치
외삽 능력            제한적          불가능           비교적 양호
적용 대상            임베딩에 더함    임베딩에 더함    Q/K에 회전 적용
파라미터 추가         없음            있음            없음
대표 모델            Transformer     BERT, GPT-2     Llama 2, VLA
                    (원래 논문)      ViT

현재 트렌드:
  소규모 모델, ViT: Learned를 여전히 많이 사용
  대규모 LLM, VLA:  RoPE가 사실상 표준
```

### 7. 위치 인코딩이 없다면?

```
위치 인코딩 제거 실험 결과 (실제 연구):

기계 번역:
  - 위치 인코딩 있음: BLEU 27.3
  - 위치 인코딩 없음: BLEU 17.5  (성능 급락!)
  → 언어에서 어순이 얼마나 중요한지를 보여줌

이미지 분류 (ViT):
  - 위치 인코딩 있음: 정확도 85.5%
  - 위치 인코딩 없음: 정확도 80.2%
  → 이미지에서도 패치의 공간 위치가 중요

VLA에서 위치 인코딩이 없다면:
  - "왼쪽으로 이동 → 집기 → 오른쪽으로 이동" 시퀀스에서
  - 순서를 인식하지 못해 동작 순서가 뒤섞임
  - 이미지에서 물체의 공간 위치를 파악하지 못함
  → 로봇이 제대로 작동할 수 없음
```

---

## 연습 주제 (Practice Topics)

스스로 생각해보고 답을 정리해 보자 (코드 작성 불필요):

1. **순열 동변성 확인**: Self-Attention의 수식 $\text{softmax}(QK^T / \sqrt{d_k})V$에서, 입력 토큰의 순서를 바꾸면(행을 셔플하면) 출력이 어떻게 변하는지 추론해보라. Q, K, V 행렬의 행이 어떻게 변하는지부터 시작하라.

2. **사인/코사인 패턴**: $d_{\text{model}}=4$인 경우, 위치 0~7에 대한 PE 값을 계산해보라 ($i=0$인 sin/cos, $i=1$인 sin/cos). 각 위치의 PE 벡터가 정말로 고유한지 확인하라.

3. **Learned vs RoPE 트레이드오프**: 학습 시 최대 시퀀스 길이가 2048이고, 추론 시 4096 길이를 처리해야 한다면, Learned Positional Encoding과 RoPE 중 어느 것이 더 적합한가? 그 이유를 설명하라.

4. **RoPE의 상대 위치 성질**: 위치 5의 토큰과 위치 8의 토큰 사이의 attention score가, 위치 100의 토큰과 위치 103의 토큰 사이의 attention score와 같아지는 이유를 RoPE의 회전 성질로 설명해보라.

5. **VLA에서의 다중 위치**: VLA의 입력 시퀀스 [이미지 패치들 | 텍스트 토큰들 | 액션 토큰들]에서, 이미지 패치의 위치와 텍스트 토큰의 위치가 연속적으로 번호가 매겨진다면, 모달리티 간 경계에서 어떤 문제가 생길 수 있을지 생각해보라.

---

## 다음 노트 (Next Note)

위치 인코딩으로 Transformer에 순서 감각을 부여했다. 이제 Self-Attention을 더 강력하게 만드는 기법인 **Multi-Head Attention**과, Transformer 블록의 나머지 절반인 **Feed-Forward Network(FFN)**를 알아보자. VLA의 Llama 2에서 사용하는 최신 FFN 기법인 SwiGLU도 다룬다.

**다음**: [Multi-Head Attention과 FFN](./05-multi-head-attention-ffn.md) - 여러 관점에서 동시에 주목하는 방법과, 수집한 정보를 처리하는 FFN. Llama 2의 SwiGLU를 포함한 완전한 Transformer Block.
