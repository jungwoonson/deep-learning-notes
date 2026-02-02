# Robot Action Spaces

## VLA 연결고리

VLA 모델의 최종 출력은 로봇 행동(action)이다. 이 action이 구체적으로 무엇을 의미하는지, 어떤 좌표계에서 정의되는지, 어떻게 정규화하는지를 이해해야 VLA의 입출력 구조를 완전히 파악할 수 있다. OpenVLA는 7차원 end-effector action을, RT-2는 discretized action token을 출력한다. 이 모든 것의 기반이 이 노트의 내용이다.

---

## 핵심 개념

### 1. Joint Space vs End-Effector Space

로봇 행동을 표현하는 두 가지 주요 좌표계가 있다.

**Joint Space (관절 공간)**:
- 각 관절의 각도(position) 또는 속도(velocity) 또는 토크(torque)를 직접 지정
- 예: 6개 관절의 각도 [q1, q2, q3, q4, q5, q6] (6-DoF 로봇 기준)
- 장점: 로봇 하드웨어에 직접 대응, 모호함 없음
- 단점: 사람이 직관적으로 이해하기 어려움

**End-Effector Space (말단장치 공간)**:
- 로봇 손(end-effector)의 위치와 자세를 지정
- 예: [x, y, z, roll, pitch, yaw, gripper]
- 장점: 직관적, 과제 관련 정보에 집중
- 단점: Inverse Kinematics(IK)가 필요, 특이점(singularity) 문제

**VLA에서의 선택**: 대부분의 VLA 모델은 **end-effector space**를 사용한다. 과제 수준의 의미가 명확하고, 다른 로봇으로의 전이(transfer)가 상대적으로 용이하기 때문이다.

### 2. 6-DoF 로봇 팔 (Fairino FR3 예시)

중국의 협동로봇(cobot) 전문 제조사 Fairino(法奥机器人)의 FR3는 6-DoF 경량 협동로봇이다. 가반하중 3kg, 작업 반경 622mm, 반복 정밀도 ±0.02mm이며, TCP/IP, Modbus TCP/RTU, EtherCAT, Profinet 통신을 지원하고 Python, C++, ROS/ROS2로 프로그래밍할 수 있다.

**6 Degrees of Freedom**:
- 6개의 회전 관절 (revolute joint): Joint 1 ~ Joint 6
- 구조: 베이스 회전(1) + 어깨(1) + 팔꿈치(1) + 손목(3)
- 6-DoF는 3차원 공간에서 위치(3) + 자세(3)를 완전히 지정하는 데 필요한 최소 자유도
- 7-DoF 로봇(예: Franka)과 달리 redundancy가 없으므로 IK 해가 유한 개(보통 최대 8개)이다

**End-Effector Action의 7차원 표현**:

| 차원 | 의미 | 범위 예시 |
|------|------|-----------|
| x | 좌우 위치 변화 | -0.05 ~ +0.05 m |
| y | 앞뒤 위치 변화 | -0.05 ~ +0.05 m |
| z | 상하 위치 변화 | -0.05 ~ +0.05 m |
| rx | x축 회전 변화 | -0.25 ~ +0.25 rad |
| ry | y축 회전 변화 | -0.25 ~ +0.25 rad |
| rz | z축 회전 변화 | -0.25 ~ +0.25 rad |
| gripper | 그리퍼 열기/닫기 | 0(열림) ~ 1(닫힘) |

### 3. Delta vs Absolute Actions

**Absolute action**: 목표 위치/자세를 절대 좌표로 지정한다.
- 예: "end-effector를 (0.5, 0.3, 0.2)로 이동하라"
- 장점: 명확함
- 단점: 초기 위치가 다르면 동일 action이 전혀 다른 결과

**Delta action**: 현재 위치로부터의 변화량을 지정한다.
- 예: "현재 위치에서 x 방향으로 +0.01m 이동하라"
- 장점: 상대적이므로 초기 위치에 덜 민감
- 단점: 오차가 누적될 수 있음

**VLA에서의 선택**: 대부분의 VLA 시스템은 **delta action**을 사용한다. OpenVLA, RT-2 모두 delta end-effector action을 출력한다. 로봇의 시작 위치가 달라도 동일한 "동작 패턴"을 학습할 수 있기 때문이다.

### 4. Action Normalization

서로 다른 로봇, 서로 다른 과제의 action을 하나의 모델로 학습하려면 정규화가 필수다.

**왜 필요한가**:
- 위치 변화량(m)과 회전 변화량(rad)의 스케일이 다르다
- 로봇마다 action 범위가 다르다
- 정규화 없이 학습하면 특정 차원이 loss를 지배한다

**일반적인 방법**:

| 방법 | 수식 | 특징 |
|------|------|------|
| Min-Max | (a - min) / (max - min) | [0, 1] 범위로 매핑 |
| Zero-mean | (a - mean) / std | 평균 0, 표준편차 1 |
| Quantile | 분위수 기반 변환 | 이상치에 강건 |

OpenVLA는 각 action 차원을 학습 데이터의 통계를 기반으로 정규화한 후, 256개 구간으로 이산화(discretize)한다.

### 5. Proprioception (자기 수용 감각)

로봇이 자신의 상태를 감지하는 감각이다.

**포함하는 정보**:
- 관절 각도 (joint positions)
- 관절 속도 (joint velocities)
- End-effector 위치/자세
- 그리퍼 상태 (열림/닫힘 정도)
- 힘/토크 센서 값

**VLA에서의 역할**:
- 카메라 이미지만으로는 로봇 자신의 정확한 상태를 알기 어렵다
- Proprioception을 추가 입력으로 제공하면 더 정확한 action 예측이 가능하다
- 일부 VLA (예: Octo, pi-0)는 proprioception을 별도 토큰으로 인코딩하여 입력한다
- 다만 OpenVLA는 이미지와 텍스트만 사용하고 proprioception은 생략한다

### 6. Action Space 설계의 현실적 고려

| 고려사항 | 설명 |
|----------|------|
| **제어 주파수** | 보통 10~50Hz. VLA는 느려서 action chunking으로 보완 |
| **Action horizon** | 한 번에 몇 스텝의 action을 예측할 것인가 (chunking) |
| **Safety limits** | Delta action의 최대 크기를 제한하여 급격한 움직임 방지 |
| **Gripper** | 연속값(얼마나 열지) vs 이산값(열기/닫기) 선택 |
| **로봇 간 호환** | End-effector space가 joint space보다 이식성 높음 |

---

## 연습 주제 (코드 없이)

1. 6-DoF 로봇 팔(Fairino FR3)에서 end-effector를 특정 위치/자세로 옮길 때, joint space solution이 여러 개(최대 8개) 존재할 수 있는 이유를 설명하라. 7-DoF 로봇(예: Franka)의 redundancy와는 어떻게 다른가?
2. Delta action의 크기가 매우 클 때 발생할 수 있는 안전 문제를 나열하라. 이를 방지하는 방법은?
3. 두 로봇(Fairino FR3와 UR5e)의 action space가 다를 때, end-effector space를 사용하면 왜 transfer가 쉬워지는지 설명하라.
4. Proprioception 없이 카메라 이미지만으로 로봇 팔의 현재 위치를 추정할 수 있는가? 어떤 한계가 있겠는가?
5. 제어 주파수가 10Hz이고 VLA의 추론 시간이 200ms라면, action chunking이 왜 필수적인지 계산해 보라.

---

## 다음 노트

[Action Tokenization](./05-action-tokenization.md)
