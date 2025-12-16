# GNSS-ACC Fusion DLL 원페이퍼

## 프로젝트 개요

GNSS(위성항법시스템)와 가속도계(ACC) 데이터를 칼만 필터로 융합하여 정밀한 변위(Displacement)를 계산하는 Windows DLL 라이브러리입니다.

**주요 특징:**
- 실시간 데이터 처리 (100Hz 기준)
- 배치 처리 지원 (100줄 단위)
- 칼만 필터 기반 센서 융합
- 단계별 검증 가능 (중간 결과 저장)

---

## 핵심 기능

### 1. 데이터 처리 파이프라인
```
입력 CSV → 파싱 → 칼만 필터 융합 → 출력 CSV
```

### 2. 칼만 필터 알고리즘
- **상태 벡터**: [위치, 속도]
- **입력**: 가속도 데이터 (Acc_Y, Acc_Z)
- **측정값**: GNSS 위치 데이터 (GPS_Y, GPS_Z)
- **출력**: 변위 (Displacement_Y, Displacement_Z)
- **파라미터**: Q (프로세스 노이즈), R (측정 노이즈) - 외부 조절 가능

### 3. 배치 처리 기능
- 100줄씩 끊어서 처리 가능
- 배치 간 칼만 필터 상태 자동 유지 (연속 처리 보장)
- 각 배치의 중간 결과를 CSV 파일로 저장하여 단계별 검증 가능

---

## API 사용법ㅡ,

### 일반 처리 모드

```c
#include "fusion_api.h"

int result = fusion_process_csv(
    "input.csv",    // 입력 파일 경로
    "output.csv",   // 출력 파일 경로
    0.1,            // Q (프로세스 노이즈 공분산)
    0.01            // R (측정 노이즈 공분산)
);
```

### 배치 처리 모드

```c
int result = fusion_process_csv_batch(
    "input.csv",    // 입력 파일 경로
    "output.csv",   // 출력 파일 경로
    0.1,            // Q
    0.01,           // R
    100,            // 배치 크기
    1               // 중간 결과 저장 (1: 저장, 0: 저장 안함)
);
```

**반환값:**
- `FUSION_SUCCESS (0)`: 성공
- 음수 값: 오류 코드 (`fusion_get_error_message()`로 메시지 확인)

---

## 데이터 형식

### 입력 CSV 형식
```
DateTime,GPS_Y,GPS_Z,Acc_Y,Acc_Z,Fix
2024-01-01 12:00:00.000,100.5,200.3,0.1,0.2,1
2024-01-01 12:00:00.010,100.6,200.4,0.15,0.25,1
...
```

**요구사항:**
- 최소 20행 이상의 데이터 필요
- DateTime 포맷: `yyyy-mm-dd HH:MM:SS.fff`
- 100Hz 기준 데이터 (0.01초 간격)

### 출력 CSV 형식
```
DateTime,Displacement_Y,Displacement_Z
2024-01-01 12:00:00.000,100.5,200.3
2024-01-01 12:00:00.010,100.6,200.4
...
```

### 중간 결과 파일 (배치 처리 시)
- `output_batch_001.csv`: 첫 번째 배치 결과
- `output_batch_002.csv`: 두 번째 배치 결과
- ...

---

## 빌드 및 실행

### 빌드 방법

```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### 테스트 프로그램 실행

**일반 모드:**
```bash
test_fusion.exe input.csv output.csv [Q] [R]
```

**배치 처리 모드:**
```bash
test_fusion.exe --batch input.csv output.csv [Q] [R] [batch_size]
```

**예시:**
```bash
# 100줄씩 배치 처리 (중간 결과 저장)
test_fusion.exe --batch input.csv output.csv 0.1 0.01 100
```

---

## 프로젝트 구조

```
fusion_dll/
├── include/
│   └── fusion_api.h          # DLL 공개 API
├── src/
│   ├── fusion_dll.cpp        # DLL 메인 로직
│   ├── csv_parser.cpp        # CSV 파싱 모듈
│   ├── kalman_filter.cpp     # 칼만 필터 구현
│   └── data_structures.h     # 데이터 구조 정의
└── test/
    └── test_fusion.cpp       # 테스트 프로그램
```

---

## 주요 기술 사항

### 칼만 필터 상태 관리
- 배치 간 상태(위치, 속도, 공분산) 자동 유지
- `getState()` / `setState()` 메서드로 상태 저장/복원 가능
- 연속 처리 보장

### 오류 처리
- 파일 존재 여부 확인
- 데이터 형식 검증
- 최소 데이터 개수 확인 (20행)
- 명확한 오류 코드 및 메시지 제공

### 성능 최적화
- 배치 단위 처리로 메모리 효율성 향상
- 실시간 처리 지원 (100Hz)

---

## 기본 파라미터 값

- **Q (프로세스 노이즈)**: 0.1 (기본값)
- **R (측정 노이즈)**: 0.01 (기본값)
- **dt (시간 간격)**: 0.01초 (고정값, 100Hz 기준)
- **배치 크기**: 100행 (기본값)

---

## 활용 예시

1. **대용량 데이터 처리**: 배치 모드로 메모리 효율적 처리
2. **단계별 검증**: 중간 결과 파일로 각 배치의 처리 결과 확인
3. **파라미터 튜닝**: Q, R 값을 조절하여 필터 성능 최적화
4. **실시간 처리**: DLL을 다른 애플리케이션에 통합하여 실시간 융합

---

**버전**: 1.0.0  
**언어**: C++17  
**플랫폼**: Windows  
**라이선스**: 프로젝트 정책에 따름

