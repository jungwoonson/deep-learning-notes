# LSTM과 GRU (Long Short-Term Memory & Gated Recurrent Unit)

## VLA 연결고리

VLA에서 로봇이 긴 작업을 수행할 때, 작업 초반의 지시("빨간 컵을 찾아")를 작업 후반까지 기억해야 한다. Vanilla RNN은 긴 시퀀스에서 과거 정보를 잃어버리지만, LSTM과 GRU는 **게이트(gate) 메커니즘**으로 어떤 정보를 기억하고 어떤 정보를 잊을지 학습한다. 이 선택적 기억 능력은 Transformer의 attention으로 발전하고, 궁극적으로 VLA가 복잡한 지시를 오랫동안 유지하며 행동을 생성하는 기반이 된다.

---

## 핵심 개념

### 1. LSTM의 핵심: Cell State

LSTM은 hidden state 외에 **cell state(셀 상태)**라는 별도의 메모리 경로를 추가했다. Cell state는 컨베이어 벨트처럼 정보를 거의 손실 없이 멀리 전달할 수 있다. 정보의 추가와 제거는 세 개의 게이트가 제어한다.

### 2. 세 가지 게이트

모든 게이트는 sigmoid 함수를 사용하여 0~1 사이의 값을 출력한다. 이 값이 정보의 통과 비율을 결정한다.

**Forget Gate (망각 게이트)**
- 질문: "cell state에서 어떤 정보를 **버릴까**?"
- 이전 hidden state와 현재 입력을 보고, cell state의 각 요소를 얼마나 유지할지 결정한다
- 출력이 0이면 완전히 잊고, 1이면 완전히 기억한다
- 예: 문장의 주어가 바뀌면, 이전 주어 정보를 잊어야 한다

**Input Gate (입력 게이트)**
- 질문: "새로운 정보 중 어떤 것을 cell state에 **저장할까**?"
- 두 부분으로 구성: (1) 어떤 값을 업데이트할지 결정하는 sigmoid, (2) 새로운 후보 값을 만드는 tanh
- 이 둘을 곱해서 cell state에 더한다

**Output Gate (출력 게이트)**
- 질문: "cell state 중 어떤 부분을 **출력할까**?"
- cell state 전체를 출력하지 않고, 현재 시점에 필요한 부분만 걸러서 hidden state로 내보낸다

### 3. LSTM 정보 흐름 요약

```
[forget gate] -> cell state에서 불필요한 정보 제거
[input gate]  -> cell state에 새로운 정보 추가
[output gate] -> cell state에서 필요한 정보를 hidden state로 출력
```

Cell state는 덧셈(addition) 연산으로 업데이트된다. 곱셈이 아닌 덧셈이기 때문에 gradient가 소멸하지 않고 멀리까지 전달될 수 있다. 이것이 LSTM이 vanishing gradient를 해결하는 핵심 원리이다.

### 4. GRU (Gated Recurrent Unit)

GRU는 LSTM을 **단순화**한 구조이다.

| 비교 항목 | LSTM | GRU |
|-----------|------|-----|
| 게이트 수 | 3개 (forget, input, output) | 2개 (reset, update) |
| 메모리 | cell state + hidden state | hidden state만 사용 |
| 파라미터 수 | 더 많음 | 더 적음 |
| 학습 속도 | 상대적으로 느림 | 상대적으로 빠름 |
| 성능 | 대체로 비슷 | 대체로 비슷 |

**Reset Gate**: 이전 hidden state를 얼마나 무시할지 결정 (LSTM의 forget gate와 유사)

**Update Gate**: 이전 hidden state와 새로운 후보 값을 어떤 비율로 섞을지 결정 (LSTM의 forget gate + input gate 역할을 하나로 통합)

GRU는 파라미터가 적어서 데이터가 적을 때 유리하고, LSTM은 복잡한 패턴을 다룰 때 유리하다. 실무에서는 둘 다 시도해보고 더 나은 것을 선택한다.

### 5. Bidirectional (양방향)

기본 RNN/LSTM/GRU는 왼쪽에서 오른쪽으로만 읽는다. 하지만 "나는 ___를 먹었다"에서 빈칸을 채우려면 뒤에 오는 "먹었다"도 알아야 한다.

**Bidirectional RNN**은 같은 시퀀스를 정방향과 역방향 두 번 처리한 뒤, 각 시점에서 두 hidden state를 합친다(concatenate). 이렇게 하면 각 시점이 과거와 미래 양쪽의 문맥을 모두 활용할 수 있다.

단, 양방향은 **전체 시퀀스가 주어졌을 때**만 사용할 수 있다. 실시간으로 한 단어씩 생성해야 하는 경우(텍스트 생성, 로봇 실시간 제어)에는 미래 정보를 알 수 없으므로 단방향만 사용한다.

### 6. Stacked (다층 구조)

RNN/LSTM/GRU를 여러 층 쌓을 수 있다. 첫 번째 층의 hidden state 시퀀스가 두 번째 층의 입력이 된다.

- 1층: 낮은 수준의 패턴 (단어 간 관계)
- 2층: 높은 수준의 패턴 (구문 구조)
- 3층: 더 추상적인 패턴 (의미)

보통 2~4층이 효과적이며, 너무 깊으면 학습이 어려워진다.

---

## 연습 주제

1. "오늘 날씨가 좋아서 기분이 ___" 문장에서 LSTM의 세 게이트가 각각 어떤 역할을 할지 직관적으로 설명해 보기
2. GRU의 update gate가 LSTM의 forget gate와 input gate를 어떻게 하나로 합쳤는지 정리하기
3. Cell state가 덧셈으로 업데이트되는 것이 왜 vanishing gradient 해결에 도움이 되는지 수학적 직관 없이 설명해 보기
4. 양방향(bidirectional) 구조를 로봇 동작 생성에 사용할 수 없는 이유를 구체적으로 서술하기
5. LSTM 2층과 GRU 3층 중 파라미터 수가 더 많은 쪽이 어디일지 추론해 보기 (정확한 계산 없이 논리적으로)

---
