"""
칼만 필터 센서 융합 - 단계별 하드코딩 버전
모든 변수를 개별적으로 초기화하고, 행렬 연산을 곱셈과 덧셈으로 명시적으로 계산
작업경로 : D:\gnss_new\preprocess\process_100Hz
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG 영역 - 사용자가 수정 가능
# ============================================================================
CONFIG = {
    # 입력 파일
    'input_csv': 'outer_merged.csv',
    'rtk_column': 'RTK_Z_01',
    
    # 출력 파일
    'output_csv': 'output_version2.csv',
    'results_csv': 'kalman_filter_results_version2.csv',
    
    # 칼만 필터 파라미터
    # ========================================================================
    # Q (프로세스 노이즈 공분산): 시스템 모델의 불확실성
    # - 단위: 상태 변수의 제곱 (위치: m^2, 속도: (m/s)^2)
    # - 의미: 가속도 적분 모델이 얼마나 불확실한지를 나타냄
    # - 값이 클수록: 모델을 덜 신뢰 (예측값에 덜 의존)
    # - 값이 작을수록: 모델을 더 신뢰 (예측값에 더 의존)
    # - 일반적인 범위: 0.001 ~ 1.0 (시스템에 따라 다름)
    # - 결정 방법: 가속도 센서의 노이즈 수준, 모델 오차를 고려
    # ========================================================================
    'Q': 0.01,  # 프로세스 노이즈 공분산 (위치와 속도 모두에 적용)
    
    # ========================================================================
    # R (측정 노이즈 공분산): 측정 센서의 노이즈
    # - 단위: 측정값의 제곱 (RTK 위치: m^2)
    # - 의미: RTK 측정값이 얼마나 노이즈가 있는지를 나타냄
    # - 값이 클수록: 측정값을 덜 신뢰 (예측값에 더 의존)
    # - 값이 작을수록: 측정값을 더 신뢰 (측정값에 더 의존)
    # - 일반적인 범위: 0.01 ~ 10.0 (센서 특성에 따라 다름)
    # - 결정 방법: RTK 센서의 정확도, 측정 오차를 고려
    #   예: RTK 정확도가 ±0.01m라면 R ≈ 0.01^2 = 0.0001
    #       RTK 정확도가 ±0.1m라면 R ≈ 0.1^2 = 0.01
    # ========================================================================
    'R': 0.1,   # 측정 노이즈 공분산 (RTK 위치 측정)
    
    # ========================================================================
    # P0 (초기 공분산 행렬): 초기 상태 추정의 불확실성
    # - P0_00: 초기 위치의 불확실성 (m^2)
    # - P0_11: 초기 속도의 불확실성 ((m/s)^2)
    # - P0_01, P0_10: 위치와 속도의 상관관계 (보통 0)
    # - 일반적으로 1.0 정도로 설정 (초기 불확실성이 크다고 가정)
    # ========================================================================
    'P0_00': 1.0,  # 초기 공분산 P[0,0] (위치)
    'P0_01': 0.0,  # 초기 공분산 P[0,1] (위치-속도 상관)
    'P0_10': 0.0,  # 초기 공분산 P[1,0] (속도-위치 상관)
    'P0_11': 1.0,  # 초기 공분산 P[1,1] (속도)
    
    # 적응형 파라미터
    'adaptive_R': True,  # RTK 신뢰도에 따라 R 조정
    'adaptive_Q': False,  # 가속도 변화에 따라 Q 조정
    
    # 가속도 스케일링 팩터
    'acc_scale_factor': 0.000002599,
}

# ============================================================================
# 데이터 로드 및 전처리
# ============================================================================
print("=" * 60)
print("칼만 필터 센서 융합 (단계별 하드코딩 버전)")
print("=" * 60)
print(f"\n데이터 로딩 중: {CONFIG['input_csv']}")

# 데이터 로드
df = pd.read_csv(CONFIG['input_csv'])
df['time'] = pd.to_datetime(df['time'])

# 시간 간격 계산
time_diffs = df['time'].diff().dt.total_seconds()
time_diffs.iloc[0] = time_diffs.iloc[1:].mean()
dt_array = time_diffs.values

# 데이터 추출
rtk_z_raw = df[CONFIG['rtk_column']].values
acc_z1 = df['Acc_Z1'].values

# RTK 유효 데이터 마스크
rtk_valid_mask = ~np.isnan(rtk_z_raw)
rtk_valid_indices = np.where(rtk_valid_mask)[0]

# RTK 데이터 보간
if len(rtk_valid_indices) > 1:
    valid_times = df['time'].values[rtk_valid_indices]
    valid_rtk = rtk_z_raw[rtk_valid_indices]
    all_times = df['time'].values
    
    interp_func = interp1d(
        valid_times.astype(np.int64), 
        valid_rtk, 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    rtk_z = interp_func(all_times.astype(np.int64))
    
    if rtk_valid_indices[0] > 0:
        rtk_z[:rtk_valid_indices[0]] = valid_rtk[0]
    if rtk_valid_indices[-1] < len(rtk_z) - 1:
        rtk_z[rtk_valid_indices[-1]+1:] = valid_rtk[-1]
else:
    rtk_z = np.full(len(rtk_z_raw), np.nanmean(rtk_z_raw) if np.any(~np.isnan(rtk_z_raw)) else 113.0)

# 가속도 스케일링 및 바이어스 제거
acc_z1_scaled = acc_z1 * CONFIG['acc_scale_factor']
acc_bias = np.mean(acc_z1_scaled)
acc_z1_scaled = acc_z1_scaled - acc_bias
 
n = len(rtk_z)
print(f"데이터 로드 완료: {n} 행")
print(f"RTK 유효 데이터: {np.sum(rtk_valid_mask)} / {n} ({np.sum(rtk_valid_mask)/n*100:.1f}%)")

# ============================================================================
# 초기값 정의
# ============================================================================
print("\n초기값 설정 중...")

# 초기 상태 (위치, 속도)
initial_rtk = rtk_z[rtk_valid_indices[0]] if len(rtk_valid_indices) > 0 else rtk_z[0]
x_position = initial_rtk
x_velocity = 0.0

# 초기 공분산 행렬 P
P_00 = CONFIG['P0_00']
P_01 = CONFIG['P0_01']
P_10 = CONFIG['P0_10']
P_11 = CONFIG['P0_11']

# 측정 행렬 H (위치만 측정)
H_00 = 1.0
H_01 = 0.0

# 프로세스 노이즈 공분산 Q
Q_base_00 = CONFIG['Q']
Q_base_01 = 0.0
Q_base_10 = 0.0
Q_base_11 = CONFIG['Q']

# 측정 노이즈 공분산 R
R_base = CONFIG['R']
R_current = R_base

# 적응형 파라미터 범위
if CONFIG['adaptive_R']:
    R_min = R_base * 0.1
    R_max = R_base * 10.0

if CONFIG['adaptive_Q']:
    Q_min_00 = Q_base_00 * 0.1
    Q_min_11 = Q_base_11 * 0.1
    Q_max_00 = Q_base_00 * 10.0
    Q_max_11 = Q_base_11 * 10.0

# 현재 Q 값
Q_current_00 = Q_base_00
Q_current_01 = 0.0
Q_current_10 = 0.0
Q_current_11 = Q_base_11

# 이전 가속도 (적응형 Q용)
prev_acc = acc_z1_scaled[0]

# 결과 저장용 배열
displacement = np.zeros(n)
velocity = np.zeros(n)
intermediate_results = []

# ============================================================================
# 초기 상태 저장 (time_step = 0)
# ============================================================================
intermediate_results.append({
    'time_step': 0,
    'time': df['time'].iloc[0],
    'dt': 0.0,
    'x_pred_position': x_position,
    'x_pred_velocity': x_velocity,
    'P_pred_00': P_00,
    'P_pred_01': P_01,
    'P_pred_10': P_10,
    'P_pred_11': P_11,
    'RTK_valid': rtk_valid_mask[0],
    'RTK_measurement': rtk_z[0] if rtk_valid_mask[0] else np.nan,
    'residual': np.nan,
    'kalman_gain_K0': np.nan,
    'kalman_gain_K1': np.nan,
    'x_updated_position': x_position,
    'x_updated_velocity': x_velocity,
    'P_updated_00': P_00,
    'P_updated_01': P_01,
    'P_updated_10': P_10,
    'P_updated_11': P_11,
    'Q_current_00': Q_current_00,
    'Q_current_11': Q_current_11,
    'R_current': R_current,
    'displacement': x_position,
    'velocity': x_velocity,
    'acceleration_input': acc_z1_scaled[0]
})

displacement[0] = x_position
velocity[0] = x_velocity

# ============================================================================
# 메인 루프 - 각 타임스텝별로 하드코딩된 계산
# ============================================================================
print("\n칼만 필터 실행 중...")
print(f"  Q = {CONFIG['Q']}, R = {CONFIG['R']}")
print(f"  Adaptive R = {CONFIG['adaptive_R']}, Adaptive Q = {CONFIG['adaptive_Q']}")

for i in range(1, n):
    # ========================================================================
    # 1. 시간 간격 계산
    # ========================================================================
    dt = dt_array[i]
    
    # ========================================================================
    # 2. 상태 전이 행렬 F 계산
    # F = [[1.0, dt], [0.0, 1.0]]
    # 
    # 물리적 의미:
    # F는 가속도 없이 시간 dt만큼 경과했을 때 상태의 변화를 나타냅니다.
    # 
    # F * [x, v]^T = [x + v*dt, v]^T
    # 
    # 즉:
    #   - 위치: x_new = x_old + v_old * dt  (속도에 의한 위치 변화)
    #   - 속도: v_new = v_old              (가속도가 없으면 속도는 변하지 않음)
    # ========================================================================
    F_00 = 1.0  # 위치 → 위치 (변화 없음)
    F_01 = dt   # 속도 → 위치 (속도 * 시간 = 위치 변화)
    F_10 = 0.0  # 위치 → 속도 (위치는 속도에 직접 영향 없음)
    F_11 = 1.0  # 속도 → 속도 (가속도 없으면 속도 유지)
    
    # ========================================================================
    # 3. 제어 입력 행렬 B 계산
    # B = [[0.5 * dt^2], [dt]]
    # 
    # 물리적 의미:
    # 등가속도 운동에서 위치 변화: Δx = v0*t + 0.5*a*t^2
    # 여기서 0.5는 적분에서 나온 계수입니다.
    # 
    # 수학적 유도:
    # v(t) = v0 + a*t  (속도)
    # x(t) = ∫v(t)dt = ∫(v0 + a*t)dt = v0*t + 0.5*a*t^2 + x0
    # 따라서: x(t) = x0 + v0*t + 0.5*a*t^2
    # 
    # B_00 = 0.5*dt^2는 가속도(a)가 위치(x)에 미치는 영향을 나타냅니다.
    # B_01 = dt는 가속도(a)가 속도(v)에 미치는 영향을 나타냅니다.
    # ========================================================================
    B_00 = 0.5 * dt * dt  # 가속도 → 위치 변환 계수 (0.5는 적분에서 나온 계수!)
    B_01 = dt             # 가속도 → 속도 변환 계수
    
    # ========================================================================
    # 4. 적응형 Q 조정 (선택적)
    # ========================================================================
    if CONFIG['adaptive_Q']:
        u_current = acc_z1_scaled[i]
        acc_change = abs(u_current - prev_acc)
        Q_scale = 1.0 + acc_change * 100.0
        
        Q_current_00 = Q_base_00 * Q_scale
        Q_current_11 = Q_base_11 * Q_scale
        
        # 클리핑
        if Q_current_00 < Q_min_00:
            Q_current_00 = Q_min_00
        elif Q_current_00 > Q_max_00:
            Q_current_00 = Q_max_00
            
        if Q_current_11 < Q_min_11:
            Q_current_11 = Q_min_11
        elif Q_current_11 > Q_max_11:
            Q_current_11 = Q_max_11
        
        prev_acc = u_current
    else:
        Q_current_00 = Q_base_00
        Q_current_11 = Q_base_11
    
    Q_current_01 = 0.0
    Q_current_10 = 0.0
    
    # ========================================================================
    # 5. 예측 단계 (Prediction)
    # x_pred = F * x + B * u
    # P_pred = F * P * F^T + Q
    # 
    # 물리적 의미:
    # 등가속도 운동 방정식을 사용하여 다음 시점의 위치와 속도를 예측합니다.
    # 
    # 위치 예측: x_pred = x0 + v0*dt + 0.5*a*dt^2
    # 속도 예측: v_pred = v0 + a*dt
    # 
    # 여기서:
    #   x0 = 현재 위치 (x_position)
    #   v0 = 현재 속도 (x_velocity)
    #   a  = 가속도 (u = acc_z1_scaled[i])
    #   dt = 시간 간격
    # 
    # 중요: 0.5 계수는 물리적으로 정확합니다!
    # 등가속도 운동에서 위치는 x = x0 + v0*t + 0.5*a*t^2 입니다.
    # 이는 속도 v(t) = v0 + a*t를 시간에 대해 적분한 결과입니다.
    # ========================================================================
    
    # 가속도 입력
    u = acc_z1_scaled[i]
    
    # 디버깅: 첫 번째 타임스텝에서 실제 사용된 값 출력
    if i == 1:
        print(f"\n[디버깅] time_step = {i}에서 사용된 값:")
        print(f"  dt = {dt:.15f}")
        print(f"  x_position (초기) = {x_position:.15f}")
        print(f"  x_velocity (초기) = {x_velocity:.15f}")
        print(f"  u (가속도, 바이어스 제거 후) = {u:.15f}")
        print(f"  B_00 = 0.5*dt^2 = {B_00:.15f}")
        print(f"  물리적 계산식: x_pred = x0 + v0*dt + 0.5*a*dt^2")
        print(f"  계산: x_pred = {x_position:.15f} + {F_01:.15f} * {x_velocity:.15f} + {B_00:.15f} * {u:.15f}")
        print(f"       = {x_position:.15f} + {F_01 * x_velocity:.15f} + {B_00 * u:.15f}")
    
    # x_pred = F * x 계산
    # F * x는 현재 상태를 다음 시점으로 전이시킵니다 (가속도 없이)
    # x_pred[0] = F[0,0]*x[0] + F[0,1]*x[1] = 1.0*x_position + dt*x_velocity
    # x_pred[1] = F[1,0]*x[0] + F[1,1]*x[1] = 0.0*x_position + 1.0*x_velocity
    x_pred_position = F_00 * x_position + F_01 * x_velocity  # x0 + v0*dt
    x_pred_velocity = F_10 * x_position + F_11 * x_velocity  # v0
    
    # x_pred = x_pred + B * u 계산
    # B * u는 가속도의 영향을 추가합니다
    # x_pred[0] += B[0,0]*u = 0.5*dt^2 * a  →  x_pred = x0 + v0*dt + 0.5*a*dt^2
    # x_pred[1] += B[1,0]*u = dt * a        →  v_pred = v0 + a*dt
    x_pred_position = x_pred_position + B_00 * u  # 최종: x0 + v0*dt + 0.5*a*dt^2
    x_pred_velocity = x_pred_velocity + B_01 * u  # 최종: v0 + a*dt
    
    # 디버깅: 첫 번째 타임스텝에서 최종 예측값 출력
    if i == 1:
        print(f"  최종 x_pred_position = {x_pred_position:.15f}")
        print(f"  물리적 계산식: x_pred = x0 + v0*dt + 0.5*a*dt^2")
        print(f"  계산식: {x_position:.15f} + {F_01:.15f} * {x_velocity:.15f} + {B_00:.15f} * {u:.15f}")
        print(f"         = {x_position:.15f} + {F_01 * x_velocity:.15f} + {B_00 * u:.15f}")
        print(f"         = {x_pred_position:.15f}")
        print(f"  참고: 0.5 계수는 등가속도 운동 방정식에서 나온 정확한 물리적 계수입니다.")
    
    # P_pred = F * P 계산
    # P_pred[0,0] = F[0,0]*P[0,0] + F[0,1]*P[1,0]
    # P_pred[0,1] = F[0,0]*P[0,1] + F[0,1]*P[1,1]
    # P_pred[1,0] = F[1,0]*P[0,0] + F[1,1]*P[1,0]
    # P_pred[1,1] = F[1,0]*P[0,1] + F[1,1]*P[1,1]
    P_pred_temp_00 = F_00 * P_00 + F_01 * P_10  # P00
    P_pred_temp_01 = F_00 * P_01 + F_01 * P_11  # P11 * dt
    P_pred_temp_10 = F_10 * P_00 + F_11 * P_10  # 0
    P_pred_temp_11 = F_10 * P_01 + F_11 * P_11  # P11
    
    # P_pred = P_pred * F^T 계산
    # F^T = [[F[0,0], F[1,0]], [F[0,1], F[1,1]]]
    # P_pred[0,0] = P_pred_temp[0,0]*F[0,0] + P_pred_temp[0,1]*F[1,0]
    # P_pred[0,1] = P_pred_temp[0,0]*F[0,1] + P_pred_temp[0,1]*F[1,1]
    # P_pred[1,0] = P_pred_temp[1,0]*F[0,0] + P_pred_temp[1,1]*F[1,0]
    # P_pred[1,1] = P_pred_temp[1,0]*F[0,1] + P_pred_temp[1,1]*F[1,1]
    F_T_00 = F_00  # F^T[0,0] = F[0,0]
    F_T_01 = F_10  # F^T[0,1] = F[1,0]
    F_T_10 = F_01  # F^T[1,0] = F[0,1]
    F_T_11 = F_11  # F^T[1,1] = F[1,1]
    
    P_pred_00 = P_pred_temp_00 * F_T_00 + P_pred_temp_01 * F_T_10
    P_pred_01 = P_pred_temp_00 * F_T_01 + P_pred_temp_01 * F_T_11
    P_pred_10 = P_pred_temp_10 * F_T_00 + P_pred_temp_11 * F_T_10
    P_pred_11 = P_pred_temp_10 * F_T_01 + P_pred_temp_11 * F_T_11
    
    # P_pred = P_pred + Q 계산
    P_pred_00 = P_pred_00 + Q_current_00
    P_pred_01 = P_pred_01 + Q_current_01
    P_pred_10 = P_pred_10 + Q_current_10
    P_pred_11 = P_pred_11 + Q_current_11
    
    # ========================================================================
    # 6. 업데이트 단계 (Update) - RTK 데이터 유효할 때만
    # ========================================================================
    if rtk_valid_mask[i]:
        # ====================================================================
        # 6-1. 적응형 R 조정 (선택적)
        # ====================================================================
        if CONFIG['adaptive_R']:
            if i > 0 and rtk_valid_mask[i-1]:
                rtk_diff = abs(rtk_z[i] - rtk_z[i-1])
                R_scale = 1.0 + rtk_diff * 10.0
                R_current = R_base * R_scale
                
                # 클리핑
                if R_current < R_min:
                    R_current = R_min
                elif R_current > R_max:
                    R_current = R_max
        else:
            R_current = R_base
        
        # ====================================================================
        # 6-2. 잔차 계산
        # y = z - H * x_pred
        # ====================================================================
        z = rtk_z[i]  # RTK 측정값
        H_x_pred = H_00 * x_pred_position + H_01 * x_pred_velocity
        residual = z - H_x_pred
        
        # ====================================================================
        # 6-3. 잔차 공분산 계산
        # S = H * P_pred * H^T + R
        # ====================================================================
        # H * P_pred 계산
        H_P_pred_0 = H_00 * P_pred_00 + H_01 * P_pred_10
        H_P_pred_1 = H_00 * P_pred_01 + H_01 * P_pred_11
        
        # H * P_pred * H^T 계산 (H^T는 [H[0,0], H[0,1]]^T = [H[0,0], H[1,0]]^T)
        H_T_0 = H_00  # H^T[0] = H[0,0]
        H_T_1 = H_01  # H^T[1] = H[0,1] (하지만 H는 1x2이므로 H^T는 2x1)
        # 실제로는 S = H * P_pred * H^T는 스칼라
        S = H_P_pred_0 * H_T_0 + H_P_pred_1 * H_T_1 + R_current
        
        # ====================================================================
        # 6-4. 칼만 게인 계산
        # K = P_pred * H^T / S
        # ====================================================================
        # P_pred * H^T 계산
        # H^T는 2x1 벡터: [H[0,0], H[0,1]]^T = [H[0,0], H[1,0]]^T
        # 하지만 H는 [1.0, 0.0]이므로 H^T = [[1.0], [0.0]]
        H_T_col0 = H_00  # H^T[0,0] = H[0,0] = 1.0
        H_T_col1 = H_01  # H^T[1,0] = H[0,1] = 0.0
        
        # K = P_pred * H^T
        K_0 = P_pred_00 * H_T_col0 + P_pred_01 * H_T_col1
        K_1 = P_pred_10 * H_T_col0 + P_pred_11 * H_T_col1
        
        # K = K / S
        K_0 = K_0 / S
        K_1 = K_1 / S
        
        # ====================================================================
        # 6-5. 상태 업데이트
        # x = x_pred + K * y
        # ====================================================================
        x_position = x_pred_position + K_0 * residual
        x_velocity = x_pred_velocity + K_1 * residual
        
        # ====================================================================
        # 6-6. 공분산 업데이트
        # P = (I - K * H) * P_pred
        # ====================================================================
        # K * H 계산
        # K는 2x1, H는 1x2이므로 K*H는 2x2
        K_H_00 = K_0 * H_00
        K_H_01 = K_0 * H_01
        K_H_10 = K_1 * H_00
        K_H_11 = K_1 * H_01
        
        # I - K * H 계산
        I_minus_KH_00 = 1.0 - K_H_00
        I_minus_KH_01 = 0.0 - K_H_01
        I_minus_KH_10 = 0.0 - K_H_10
        I_minus_KH_11 = 1.0 - K_H_11
        
        # (I - K * H) * P_pred 계산
        P_00 = I_minus_KH_00 * P_pred_00 + I_minus_KH_01 * P_pred_10
        P_01 = I_minus_KH_00 * P_pred_01 + I_minus_KH_01 * P_pred_11
        P_10 = I_minus_KH_10 * P_pred_00 + I_minus_KH_11 * P_pred_10
        P_11 = I_minus_KH_10 * P_pred_01 + I_minus_KH_11 * P_pred_11
        
        # ====================================================================
        # 중간 결과 저장 (RTK 유효)
        # ====================================================================
        intermediate_results.append({
            'time_step': i,
            'time': df['time'].iloc[i],
            'dt': dt,
            'x_pred_position': x_pred_position,
            'x_pred_velocity': x_pred_velocity,
            'P_pred_00': P_pred_00,
            'P_pred_01': P_pred_01,
            'P_pred_10': P_pred_10,
            'P_pred_11': P_pred_11,
            'RTK_valid': True,
            'RTK_measurement': z,
            'residual': residual,
            'kalman_gain_K0': K_0,
            'kalman_gain_K1': K_1,
            'x_updated_position': x_position,
            'x_updated_velocity': x_velocity,
            'P_updated_00': P_00,
            'P_updated_01': P_01,
            'P_updated_10': P_10,
            'P_updated_11': P_11,
            'Q_current_00': Q_current_00,
            'Q_current_11': Q_current_11,
            'R_current': R_current,
            'displacement': x_position,
            'velocity': x_velocity,
            'acceleration_input': u
        })
    else:
        # ====================================================================
        # RTK 데이터가 없으면 예측값만 사용
        # ====================================================================
        x_position = x_pred_position
        x_velocity = x_pred_velocity
        P_00 = P_pred_00
        P_01 = P_pred_01
        P_10 = P_pred_10
        P_11 = P_pred_11
        
        # ====================================================================
        # 중간 결과 저장 (RTK 무효)
        # ====================================================================
        intermediate_results.append({
            'time_step': i,
            'time': df['time'].iloc[i],
            'dt': dt,
            'x_pred_position': x_pred_position,
            'x_pred_velocity': x_pred_velocity,
            'P_pred_00': P_pred_00,
            'P_pred_01': P_pred_01,
            'P_pred_10': P_pred_10,
            'P_pred_11': P_pred_11,
            'RTK_valid': False,
            'RTK_measurement': np.nan,
            'residual': np.nan,
            'kalman_gain_K0': np.nan,
            'kalman_gain_K1': np.nan,
            'x_updated_position': x_position,
            'x_updated_velocity': x_velocity,
            'P_updated_00': P_00,
            'P_updated_01': P_01,
            'P_updated_10': P_10,
            'P_updated_11': P_11,
            'Q_current_00': Q_current_00,
            'Q_current_11': Q_current_11,
            'R_current': R_current,
            'displacement': x_position,
            'velocity': x_velocity,
            'acceleration_input': u
        })
    
    # 최종 결과 저장
    displacement[i] = x_position
    velocity[i] = x_velocity

print("완료")

# ============================================================================
# 결과 저장
# ============================================================================
print(f"\n결과 저장 중...")

# 중간 결과 저장 (output.csv)
intermediate_df = pd.DataFrame(intermediate_results)
intermediate_df.to_csv(CONFIG['output_csv'], index=False)
print(f"중간 결과가 '{CONFIG['output_csv']}'에 저장되었습니다.")
print(f"  총 {len(intermediate_df)} 개의 타임스텝 데이터가 저장되었습니다.")

# 최종 결과 저장
results_df = pd.DataFrame({
    'time': df['time'].values,
    'RTK_Z_Raw': rtk_z_raw,
    'RTK_Z_Interpolated': rtk_z,
    'Displacement_Estimated': displacement,
    'Velocity_Estimated': velocity,
    'Acc_Z1': acc_z1,
    'Acc_Z1_Scaled': acc_z1_scaled,
    'RTK_Valid': rtk_valid_mask
})
results_df.to_csv(CONFIG['results_csv'], index=False)
print(f"최종 결과가 '{CONFIG['results_csv']}'에 저장되었습니다.")

print("\n" + "=" * 60)
print("작업 완료!")
print("=" * 60)

# ============================================================================
# Q와 R 튜닝 가이드
# ============================================================================
"""
Q와 R 값 결정 방법:

1. Q (프로세스 노이즈) 결정:
   - 가속도 센서의 노이즈 수준을 고려
   - 가속도 노이즈가 크면 Q를 크게 설정
   - 일반적으로 0.001 ~ 0.1 범위에서 시작
   - 튜닝: Q를 크게 하면 예측값에 덜 의존, 작게 하면 예측값에 더 의존

2. R (측정 노이즈) 결정:
   - RTK 센서의 정확도를 고려
   - RTK 정확도가 ±σ m라면 R ≈ σ^2
   - 예: RTK 정확도 ±0.01m → R ≈ 0.0001
   - 예: RTK 정확도 ±0.1m → R ≈ 0.01
   - 일반적으로 0.01 ~ 1.0 범위에서 시작
   - 튜닝: R을 크게 하면 측정값에 덜 의존, 작게 하면 측정값에 더 의존

3. Q와 R의 상대적 크기:
   - Q/R 비율이 중요함
   - Q/R이 크면: 예측값에 더 의존 (측정값을 덜 신뢰)
   - Q/R이 작으면: 측정값에 더 의존 (예측값을 덜 신뢰)
   - 일반적으로 Q < R (측정값이 예측값보다 신뢰할 만함)

4. 튜닝 절차:
   a) 초기값 설정: Q=0.01, R=0.1 (현재 설정)
   b) 결과 확인: output.csv에서 residual, error 확인
   c) Q 조정: 
      - 예측값이 측정값보다 크게 벗어나면 Q 증가
      - 예측값이 측정값을 잘 따르면 Q 감소
   d) R 조정:
      - 측정값이 노이즈가 많으면 R 증가
      - 측정값이 안정적이면 R 감소
   e) 반복: 성능 지표(MAE, RMSE)를 확인하며 최적값 찾기

5. 적응형 파라미터 사용:
   - adaptive_R=True: RTK 신뢰도에 따라 R 자동 조정
   - adaptive_Q=True: 가속도 변화에 따라 Q 자동 조정
   - 적응형을 사용하면 초기 Q, R 값의 정확도가 덜 중요함
"""

