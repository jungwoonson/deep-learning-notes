# Fairino FR3 배포: 실제 로봇에 VLA 탑재 (Deployment on Fairino FR3)

## VLA와의 연결

**이 노트는 전체 커리큘럼의 최종 장이다.** 수학 기초부터 시작하여 딥러닝, CNN, Transformer, LLM, 멀티모달, RL, Diffusion을 거쳐, VLA의 이론과 학습 방법을 배웠다. 이 마지막 노트에서는 그 모든 지식을 실제 로봇 -- Fairino FR3 -- 에 배포하는 전체 파이프라인을 다룬다. 이론에서 현실로의 마지막 다리이며, "카메라 이미지 + 자연어 명령 → 로봇 동작"이라는 VLA의 약속을 실현하는 과정이다.

---

## 핵심 개념

### 1. Fairino FR3 소개

```
FAIRINO FR3 (法奥机器人 FR3):

종류:     6-DoF 협동 로봇 팔 (collaborative robot arm)
DoF:      6 (관절 6개, 각각 회전)
도달 범위: 622mm
하중:     3kg (엔드이펙터 포함)
반복 정밀도: ±0.02mm (매우 높은 정밀도)
무게:     ~15kg
최대 TCP 속도: 1 m/s
보호 등급: IP54/IP65

통신 인터페이스:
  - TCP/IP
  - Modbus TCP/RTU
  - EtherCAT
  - Profinet

프로그래밍:
  - Python (Fairino SDK)
  - C++ (Fairino SDK)
  - ROS/ROS2 (Fairino ROS2 드라이버)
  - Teach Pendant (10.1" 터치스크린) 또는 웹 앱

컨트롤러: 내장형 컨트롤러 (별도 제어 PC 필수 아님)

왜 Fairino FR3인가:
  1. 가성비: Franka FR3 대비 훨씬 저렴한 가격대
     → 연구실/스타트업에서 접근성 높음
  2. 6-DoF: 대부분의 VLA task에 충분한 자유도
     → pick-and-place, 서랍 열기, 물체 분류 등
  3. 높은 정밀도: ±0.02mm 반복 정밀도
     → Franka (±0.1mm) 대비 5배 정밀
  4. ROS2 지원: 표준 ROS2 생태계와 호환
     → MoveIt2, ros2_control 등 활용 가능
  5. 내장 컨트롤러: 별도 실시간 PC 불필요
     → PREEMPT_RT 커널 설정 등의 번거로움 없음
  6. 다양한 통신: EtherCAT, Modbus 등 산업 표준 지원

비용:
  Fairino FR3 본체: Franka ($30,000-40,000) 대비 훨씬 저렴
    → 정확한 가격은 대리점 문의 필요
    → 중국 코봇(FAIRINO, AUBO, Elite 등)은
       일반적으로 해외 제품의 1/3-1/2 가격대
  그리퍼: 별도 구매 (Fairino 그리퍼 또는 외장 그리퍼)
  → 저가 대안: SO-100 (~$300), Koch v1.1 (~$500)
```

### 2. 전체 배포 파이프라인 개요

```
VLA on Fairino FR3: 전체 파이프라인

Phase 1: 하드웨어 설정
  Fairino FR3 로봇 + 카메라 + GPU 서버 + 네트워크

Phase 2: 데이터 수집
  텔레오퍼레이션으로 시연 데이터 수집

Phase 3: 데이터 포맷팅
  수집 데이터 → RLDS 또는 LeRobot 형식 변환

Phase 4: 모델 선택 및 파인튜닝
  SmolVLA 또는 OpenVLA → LoRA 파인튜닝

Phase 5: 추론 설정
  양자화, 서빙 환경, 통신 설정

Phase 6: 실시간 제어 루프
  카메라 → VLA → 행동 → Fairino FR3 관절 명령

Phase 7: 안전 및 모니터링
  충돌 방지, 비상 정지, 로깅
```

### 3. Phase 1: 하드웨어 설정

```
필요 하드웨어:

1. Fairino FR3 로봇
   - 내장 컨트롤러 전원 ON, 네트워크 연결
   - Teach Pendant 또는 웹 앱에서 초기 설정
   - Fairino SDK (Python/C++) 설치
   - ROS2 드라이버 설치 (fairino_ros2)
   - 별도 실시간 커널(PREEMPT_RT) 불필요
     → Fairino는 내장 컨트롤러가 실시간 처리 담당

2. 카메라 시스템
   옵션 A: 외부 카메라 (추천)
     Intel RealSense D435/D455
     - RGB + Depth (깊이 정보)
     - 30-60 FPS, 640x480 또는 1280x720
     - Fairino FR3 작업 공간이 잘 보이는 위치에 고정
   옵션 B: 손목 카메라 (wrist-mounted)
     소형 카메라를 엔드이펙터 근처에 부착
     - 가까운 물체에 대한 상세 정보
     - 외부 카메라와 병행 사용 권장

3. GPU 서버
   추론 전용: RTX 4090 (24GB) 또는 A100 (40/80GB)
   학습 + 추론: A100 (80GB) 또는 H100 권장
   최소 RAM: 64GB
   스토리지: SSD 1TB+ (데이터셋 + 모델 저장)

   참고: 140GB GPU 서버는 A100 80GB x 2 또는
   H100 80GB x 2 환경을 의미 (전체 VRAM 합산)

4. 네트워크 연결
   Fairino FR3 ↔ GPU 서버/제어 노드: Ethernet (TCP/IP)
     → Fairino는 내장 컨트롤러를 통해 직접 네트워크 통신
     → 별도 제어 PC 없이도 SDK로 직접 명령 전송 가능
   카메라 ↔ GPU 서버: USB 3.0 또는 Ethernet
   실시간성이 요구되면 EtherCAT 인터페이스 활용
```

### 4. Phase 2: 데이터 수집 (텔레오퍼레이션)

```
Fairino FR3 텔레오퍼레이션 방법:

방법 A: 직접 교시 (Direct Teaching / Gravity Compensation)
  Fairino FR3의 직접 교시 모드 활성화
  → 사람이 로봇 팔을 직접 잡고 움직임
  → 내장 중력 보상 기능으로 가볍게 조작 가능
  → 관절 위치가 자동 기록
  장점: 가장 직관적, 추가 장비 불필요
  단점: 그리퍼 제어가 별도 (버튼/페달)
  설정: Teach Pendant에서 직접 교시 모드 진입

방법 B: 스페이스마우스 (3Dconnexion SpaceMouse)
  6-DoF 입력 장치로 엔드이펙터 제어
  - x, y, z 이동 + roll, pitch, yaw 회전
  - 별도 버튼으로 그리퍼 제어
  장점: 정밀한 제어, 일정한 속도
  단점: 학습 곡선, 직관성 떨어짐

방법 C: VR 텔레오퍼레이션
  Meta Quest + 커스텀 매핑
  장점: 6-DoF + 손가락 제어, 몰입감
  단점: 설정 복잡, VR 하드웨어 필요

데이터 수집 프로토콜:
  1. task 정의 (예: "테이블 위의 컵을 집어서 선반에 놓기")
  2. 물체 배치 다양화 (매 에피소드마다 위치/방향 변경)
  3. 100-200 에피소드 수집 목표
  4. 수집 중 기록:
     - 카메라 이미지 (30Hz)
     - 관절 상태 (Fairino SDK에서 읽은 6 관절 위치/속도)
     - 엔드이펙터 위치/자세
     - 그리퍼 상태
     - 자연어 task 설명
  5. 실패 에피소드 즉시 표시 (나중에 필터링)
```

### 5. Phase 3: 데이터 포맷팅

```
수집 데이터 → VLA 학습 형식 변환:

Fairino FR3에서의 원시 데이터:
  - ROS bag (rosbag2) 또는 HDF5 파일
  - 이미지: sensor_msgs/Image
  - 관절: sensor_msgs/JointState (6개 관절)
  - 엔드이펙터: geometry_msgs/Pose
  - 그리퍼: std_msgs/Float64 또는 커스텀 메시지

SmolVLA용 (LeRobot 형식) 변환:
  1. 이미지 추출: ROS bag에서 이미지를 프레임별 추출
  2. 행동 계산:
     joint position 차이 (delta) 또는 절대 position
     → Fairino FR3의 경우: 6 관절 + 1 그리퍼 = 7차원 행동
  3. 시간 정렬: 이미지와 행동의 타임스탬프 동기화
  4. 정규화 통계 계산: 각 차원의 mean, std
  5. LeRobot Parquet 형식으로 저장

OpenVLA용 (RLDS 형식) 변환:
  1. 이미지: 224x224로 리사이즈
  2. 행동: 7차원 delta (x, y, z, rx, ry, rz, gripper)
     관절 공간 → 데카르트 공간 변환 필요
  3. TFRecord로 직렬화
  4. 에피소드별 분리

주의사항:
  - Fairino FR3의 관절 행동과 데카르트 행동의 차이
    관절 행동: [q1, q2, q3, q4, q5, q6] (6 관절각)
    데카르트 행동: [x, y, z, rx, ry, rz] + gripper
    → VLA 모델에 따라 적합한 형태 선택
    → SmolVLA: 관절 행동 직접 사용 가능
    → OpenVLA: 데카르트 행동 권장 (사전학습 형식)
  - 6-DoF이므로 관절 행동과 데카르트 행동이 동일한
    차원(6)을 가짐 → 변환이 더 직관적
    (7-DoF는 여유 자유도가 있어 역운동학 해가 무한)
```

### 6. Phase 4: 모델 선택 및 파인튜닝

```
Fairino FR3에 적합한 VLA 모델 선택:

SmolVLA (450M) - 추천 시작점:
  장점:
    - 빠른 파인튜닝 (2-4시간)
    - RTX 4090으로 충분
    - Flow Matching으로 부드러운 궤적
    - LeRobot 생태계 완전 통합
  단점:
    - 복잡한 추론 능력 제한
    - 새로운 물체 일반화 약함
  적합: 특정 task에 특화, 빠른 프로토타이핑

  Fairino FR3 적용 시 참고:
    - action 차원을 6+1=7로 설정 (6 관절 + 그리퍼)
    - 140GB VRAM 서버에서는 여유롭게 파인튜닝 가능
    - 여러 task를 동시에 학습하는 멀티태스크도 가능

OpenVLA (7B) - 높은 성능 필요 시:
  장점:
    - 강한 추론 능력 (VLM 지식)
    - 새로운 물체 일반화 우수
    - 복잡한 언어 명령 이해
  단점:
    - 파인튜닝에 A100+ 필요
    - 추론 속도 제한 (~5-10Hz)
    - 이산 토큰의 정밀도 한계
  적합: 다양한 물체/task, 높은 일반화 요구

  Fairino FR3 적용 시 참고:
    - action 차원: 7 (데카르트 6 + 그리퍼 1)
    - 140GB VRAM 서버에서 LoRA 파인튜닝 충분
    - 풀 파인튜닝도 가능한 VRAM 여유

추천 전략:
  1단계: SmolVLA로 빠르게 파인튜닝하여 파이프라인 검증
  2단계: 성능이 부족하면 OpenVLA로 전환
  3단계: 두 모델의 결과를 비교하여 최적 선택
```

### 7. Phase 5-6: 추론 및 실시간 제어 루프

```
실시간 제어 루프 아키텍처:
```

```mermaid
graph TD
    cam["카메라"] -->|"이미지 (30Hz)"| gpu

    subgraph gpu["GPU 서버 (VLA 추론)"]
        inp["이미지 + 명령"] --> vla["SmolVLA / OpenVLA"]
        vla --> chunk["Action Chunk (16 스텝)"]
    end

    gpu -->|"행동 전송\n(TCP/IP or EtherCAT)"| ctrl

    subgraph ctrl["Fairino FR3 내장 컨트롤러"]
        recv["Action Chunk 수신"] --> interp["보간: chunk → 제어"]
        interp --> safety["안전 검사\n(관절 한계, 속도 제한)"]
        safety --> exec["Fairino SDK\n→ 관절 명령 실행"]
    end

제어 인터페이스 옵션:
  A. Fairino Python SDK (TCP/IP):
     - 가장 간단한 설정
     - SDK 함수 호출로 관절 위치/속도 명령
     - 제어 주기: ~50-100Hz 수준
     - 프로토타이핑에 적합

  B. ROS2 인터페이스 (fairino_ros2):
     - ros2_control 프레임워크 활용
     - JointTrajectoryController 사용
     - MoveIt2와 연동 가능
     - 제어 주기: ~100-200Hz

  C. EtherCAT (실시간 필요 시):
     - 산업용 실시간 통신
     - 더 높은 제어 주기 가능
     - 설정이 복잡하지만 성능 최고

타이밍 예시 (SmolVLA):
  VLA 추론: 50-100ms (10-20Hz)
  Action Chunk: 16 스텝 × 33ms = 533ms
  → 추론 완료 전에 이전 chunk 실행
  → 비동기 실행으로 끊김 없는 동작

타이밍 예시 (OpenVLA):
  VLA 추론: 100-200ms (5-10Hz)
  Action Chunk: 더 짧게 (또는 단일 스텝)
  → 제어 빈도가 낮아 빠른 동작에 불리
```

### 8. Phase 7: 안전 고려사항

```
로봇 안전은 최우선 사항:

1. 하드웨어 안전
   - 비상 정지 버튼 (E-Stop): 항상 접근 가능한 위치에
   - Fairino FR3 내장 안전 기능:
     → 충돌 감지: 내장 센서 기반 자동 정지
     → 속도 제한: 컨트롤러에서 TCP 속도 상한 설정
     → 안전 영역 설정: 소프트웨어로 동작 범위 제한
     → IP54/IP65 보호 등급으로 분진/물에 대한 내성
   - 작업 공간 제한: Teach Pendant에서 안전 영역 설정
     예: z > 0.05m (테이블 아래로 가지 않도록)

2. 소프트웨어 안전
   - 관절 한계 검사: 각 관절의 위치/속도/가속도 한계
     → Fairino 컨트롤러가 내부적으로 한계 검사 수행
     → SDK 레벨에서도 추가 검사 권장
   - 행동 클리핑(clipping): VLA 출력을 안전 범위로 제한
     예: delta_position을 ±5cm로 제한
   - Watchdog 타이머: VLA 추론이 500ms 내 응답 없으면 정지
   - Smoothing: 급격한 행동 변화 방지
     Exponential Moving Average (EMA) 적용

3. 운영 안전
   - 최초 실행 시 저속 모드 (속도 10%로 제한)
   - 사람이 항상 로봇 옆에서 관찰
   - 새로운 task는 빈 테이블에서 먼저 테스트
   - 깨지기 쉬운 물체는 나중에 도입

4. VLA 특유의 안전 위험
   - Hallucination: VLA가 없는 물체를 향해 행동
     → 정기적 이미지 확인, 행동 범위 제한
   - Distribution Shift: 학습과 다른 환경에서 예측 불가한 행동
     → 초기에 보수적 설정, 점진적 확장
   - Action Chunk 중간 수정: chunk 실행 중 환경 변화
     → chunk 크기를 작게 시작, 재계획 빈도 높이기
```

### 9. Gemini Robotics On-Device: Fairino FR3 적용

```
Gemini Robotics On-Device (Google DeepMind, 2025):

Google의 접근:
  Gemini (VLM) → 로봇 제어에 특화된 On-Device 버전
  → 로봇 내장 컴퓨터에서 실행 가능한 경량 모델

Fairino FR3에 적용할 때의 시사점:
  1. On-Device 추론의 중요성
     클라우드 의존 → 네트워크 지연, 프라이버시 문제
     On-Device → 저지연, 독립적 동작
     Fairino FR3 + GPU 서버: 로컬 On-Device와 유사한 효과
     → Fairino의 내장 컨트롤러는 실시간 모터 제어 담당
     → GPU 서버는 VLA 추론 전담 → 역할 분리

  2. VLM 기반 안전 판단
     Gemini가 "이 행동이 안전한가?"를 판단
     → Fairino FR3에서도 VLA 출력을 VLM이 검증하는 구조 가능

  3. 자연어 인터랙션
     사용자가 실시간으로 명령 수정
     "아, 다른 컵이야" → VLA가 즉시 목표 변경
     → Gemini의 강력한 언어 이해력 활용

Fairino FR3에 적용하는 방법:
  SmolVLA를 기본으로 사용하되,
  Gemini API를 고수준 계획(planning)에 활용하는 하이브리드 구조
  → Dual-System 아키텍처의 변형:
     System 2: Gemini API (클라우드, 복잡한 추론)
     System 1: SmolVLA (로컬 GPU, 빠른 행동 생성)
     Fairino FR3: 최종 실행 (내장 컨트롤러의 안전 보장)
```

---

## 커리큘럼 전체 요약

```
이 커리큘럼에서 배운 것:

Part 01. 수학 기초
  → 벡터, 행렬, 미분, 확률 → 모든 AI의 언어

Part 02. Python 기초
  → 구현의 도구

Part 03. NumPy & 데이터
  → 데이터 처리의 기반

Part 04. ML 기초
  → 회귀, 분류, 손실함수, 최적화

Part 05. PyTorch
  → 텐서, 자동미분, 학습 루프

Part 06. 신경망
  → MLP, 활성화, 역전파, 정규화

Part 07. CNN
  → 합성곱, ResNet, 전이학습
  → VLA의 비전 인코더(DINOv2, SigLIP)의 기반

Part 08. 시퀀스 모델
  → RNN, LSTM, 임베딩
  → 자기회귀 생성의 기원

Part 09. Attention & Transformer
  → Self-Attention, Multi-Head, Transformer 구조
  → VLA의 핵심 아키텍처

Part 10. LLM
  → GPT, Llama, 토큰화, LoRA, RLHF
  → VLA의 언어 백본 (Llama 2)

Part 11. Modern Vision
  → ViT, DINOv2, CLIP
  → VLA의 비전 인코더

Part 12. 멀티모달
  → VLM, 이미지-텍스트 정렬, Projector
  → VLA의 직접적 기반 (VLM → VLA)

Part 13. RL & 모방학습
  → 강화학습, 행동 복제, 보상 설계
  → VLA 학습의 기본 방법론 + pi-star-0.6의 RL

Part 14. Diffusion & Flow
  → DDPM, Flow Matching
  → pi-0, SmolVLA의 행동 생성 방식

Part 15. VLA [현재 파트]
  → VLA 개요, RT-1/RT-2, OpenVLA, pi-0, Dual-System,
     SmolVLA, Action Representations, Datasets,
     LeRobot 파인튜닝, Fairino FR3 배포

모든 것이 연결된다:
  수학 → ML → 딥러닝 → Transformer → LLM → VLM → VLA → 로봇
```

---

## 미래 방향 (2026년 이후)

```
VLA 연구의 향후 방향:

1. World Models (세계 모델)
   로봇이 "행동의 결과를 상상"할 수 있는 모델
   → "이렇게 밀면 컵이 떨어질 것이다" 예측
   → NVIDIA Cosmos, Genie 2 등
   → VLA + World Model = 더 안전하고 효율적인 계획

2. Sim-to-Real Transfer
   시뮬레이션에서 학습 → 실제 로봇에 전이
   → 데이터 수집 비용 대폭 감소
   → GEN-0, NVIDIA Isaac Sim 활용
   → Domain Randomization, Domain Adaptation

3. Multi-Robot Collaboration
   여러 로봇이 협업하여 복잡한 task 수행
   → VLA가 다른 로봇의 상태도 고려
   → 분산 Dual-System 아키텍처

4. Lifelong Learning
   로봇이 배포 후에도 지속적으로 학습
   → pi-star-0.6의 RECAP 확장
   → 새로운 환경/task에 자동 적응
   → Catastrophic Forgetting 방지

5. Safety & Alignment
   VLA의 안전한 행동 보장
   → RLHF의 로봇 버전 (인간 선호도에서 학습)
   → 물리 법칙 준수 (물리 기반 제약)
   → 윤리적 행동 가이드라인

6. Embodied Foundation Models
   하나의 모델이 모든 로봇을 제어
   → RT-2의 비전을 진정으로 실현
   → 수십 종 로봇, 수천 task
   → "로봇의 GPT-4 모먼트"

7. Humanoid Revolution
   Figure, Tesla Optimus, 1X, Unitree 등
   VLA + 휴머노이드 → 범용 로봇 에이전트
   → 2026-2030년 가장 뜨거운 분야

이 커리큘럼을 마친 당신은:
  - VLA의 이론적 기반을 갖추었다
  - 주요 모델의 구조와 차이를 이해한다
  - 파인튜닝과 배포의 실질적 과정을 안다
  - 향후 연구 방향을 파악하고 있다

→ VLA 연구/개발에 참여할 준비가 되었다!
```

---

## 연습 주제 (코드 없이 생각해보기)

1. **파이프라인 설계**: Fairino FR3에서 "다양한 과일을 분류하여 바구니에 넣기" task를 수행하기 위한 전체 파이프라인을 설계하라. 데이터 수집 전략, 모델 선택, 학습 설정, 안전 설정을 포함하라. Fairino의 6-DoF 특성과 ±0.02mm 정밀도를 어떻게 활용할 수 있는가?

2. **SmolVLA vs OpenVLA on Fairino FR3**: Fairino FR3에서 "서랍 열기" task에 SmolVLA와 OpenVLA를 각각 사용할 때의 장단점을 비교하라. 제어 주파수, 일반화, 학습 비용, 하드웨어 요구를 모두 고려하라. 6-DoF에서의 action 차원(7차원 = 6 관절 + 그리퍼)이 모델 선택에 어떤 영향을 주는가?

3. **안전 시나리오**: VLA가 예측하지 못한 물체(예: 고양이)가 작업 공간에 들어왔을 때 어떤 일이 발생할 수 있는가? Fairino FR3의 내장 충돌 감지 기능과 VLA 레벨의 안전 장치를 결합하여 3단계 안전 메커니즘을 설계하라.

4. **비용 분석**: Fairino FR3 기반 VLA 시스템의 총 비용(로봇, GPU, 카메라, 소프트웨어)을 추정하라. Franka FR3 기반 시스템, SO-100 기반 시스템과 각각 비교하면 어떤가? Fairino FR3는 가격 대비 성능에서 어떤 위치에 있는가?

5. **미래 예측**: 2028년의 VLA는 어떤 모습일까? 모델 크기, 데이터 규모, 지원 로봇 수, 성공률, 가격 측면에서 예측해보라. 이 커리큘럼의 어떤 부분이 가장 빨리 변할 것인가?

6. **커리큘럼 회고**: Part 1(수학)부터 Part 15(VLA)까지를 돌아보며, 각 파트가 VLA 이해에 어떻게 기여했는지를 한 문장씩 정리하라. 가장 중요했던 파트는 무엇이고, 추가로 배우고 싶은 주제는?

---

## 커리큘럼을 마치며

이 커리큘럼은 완전 초보자가 VLA를 이해하기까지의 전체 경로를 제시했다. 수학적 기초부터 최신 2026년 연구까지, 각 단계가 다음 단계의 기반이 되는 구조로 설계되었다.

VLA는 아직 초기 단계이다. 2026년 현재, 가장 좋은 VLA도 인간 수준에는 크게 못 미친다. 그러나 발전 속도는 LLM의 궤적을 따르고 있으며, 로봇 AI의 "ChatGPT 모먼트"가 다가오고 있다.

이 분야에서 가장 중요한 것은 기초 체력이다. Transformer를 이해하지 못하면 VLA를 이해할 수 없고, 선형대수를 모르면 Transformer를 이해할 수 없다. 이 커리큘럼은 그 기초 체력을 쌓기 위해 존재한다.

다음 단계로 추천하는 것:
- SmolVLA를 직접 실행해보기 (Hugging Face Hub)
- LeRobot 튜토리얼 따라하기
- Open X-Embodiment 데이터셋 탐색하기
- 최신 VLA 논문 읽기 (arXiv robotics)
- Fairino FR3에 Fairino SDK + ROS2 드라이버를 설치하고 기본 동작 테스트하기
- 저가 로봇(SO-100)으로 병렬 실습 환경 구축하기
