"""
GNSS 변위 추정 시스템 - 파이썬 구현
원본 C++ DLL 코드를 파이썬으로 포팅
"""

import numpy as np
import pandas as pd
from scipy import signal
import os
import sys
from typing import Tuple, List, Optional

# 상수 정의
DT = 0.01  # 샘플링 시간 간격
MAX_LINE = 100  # 최대 처리 라인 수


class ButterworthFilter:
    """Butterworth 필터 클래스 - C++의 Butterworth 필터링 기능"""
    
    def __init__(self):
        # 저역 통과 필터 계수 (Mdata.cpp 100-127라인 참조)
        self.gain = 779.6778047
        self.butter_low_acc_in = np.zeros((6, 3))  # x, y, z 각각
        self.butter_low_acc_out = np.zeros((6, 3))
        
        # 고역 통과 필터 계수 (Mdata.cpp 130-183라인 참조)
        self.butter_high_gps_in = np.zeros((5, 3))
        self.butter_high_gps_out = np.zeros((5, 3))
        
        # Butterworth 고역 통과 필터 계수
        self.a = np.array([1, -3.975357288815810, 5.926375044184357, -3.926676041557190, 0.975658294000916])
        self.b = np.array([4.882670934236444e-10, 1.953068373694578e-9, 2.929602560541866e-9, 1.953068373694578e-9, 4.882670934236444e-10])
    
    def butterworth_lowpass_acc(self, acc_x, acc_y, acc_z):
        """
        가속도계 데이터에 Butterworth 저역 통과 필터 적용
        Mdata.cpp 100-127라인 참조
        """
        # 큐 시프트
        for i in range(5):
            self.butter_low_acc_in[i, :] = self.butter_low_acc_in[i+1, :]
            self.butter_low_acc_out[i, :] = self.butter_low_acc_out[i+1, :]
        
        # 새로운 입력 추가
        self.butter_low_acc_in[5, 0] = acc_x / self.gain
        self.butter_low_acc_in[5, 1] = acc_y / self.gain
        self.butter_low_acc_in[5, 2] = acc_z / self.gain
        
        # FIR 필터 계수 적용 (IIR 형태)
        filtered_acc = np.zeros(3)
        
        for axis in range(3):
            filtered_acc[axis] = (
                (self.butter_low_acc_in[0, axis] + self.butter_low_acc_in[5, axis]) +
                5 * (self.butter_low_acc_in[1, axis] + self.butter_low_acc_in[4, axis]) +
                10 * (self.butter_low_acc_in[2, axis] + self.butter_low_acc_in[3, axis]) +
                0.1254306222 * self.butter_low_acc_out[0, axis] +
                -0.8811300754 * self.butter_low_acc_out[1, axis] +
                2.5452528683 * self.butter_low_acc_out[2, axis] +
                -3.8060181193 * self.butter_low_acc_out[3, axis] +
                2.9754221097 * self.butter_low_acc_out[4, axis]
            )
            self.butter_low_acc_out[5, axis] = filtered_acc[axis]
        
        return filtered_acc


class HDRProcessor:
    """HDR (Heritage Dynamic Record) 처리 클래스"""
    
    def __init__(self):
        self.hdr_out = np.zeros((2, 3))  # 이전/현재 출력
        self.hdr_I = np.array([0.015, 0.015, 0.015])  # 적분 초기값
        self.hdr_thres = np.array([20.0, 20.0, 20.0])
        self.hdr_c1 = np.array([1.3, 1.3, 1.3])
        self.hdr_c2 = np.array([0.2, 0.2, 0.2])
        self.hdr_r = np.array([1.0, 1.0, 1.0])
        self.hdr_R = np.array([1.0, 1.0, 1.0])
        self.hdr_ic = 0.015
    
    def get_sign(self, val):
        """부호 반환"""
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0
    
    def process(self, pos_rtk, acc):
        """
        HDR 처리 - Mdata.cpp 186-252라인 참조
        """
        output = np.zeros(3)
        
        for axis in range(3):
            # 출력 업데이트
            self.hdr_out[0, axis] = self.hdr_out[1, axis]
            self.hdr_out[1, axis] = pos_rtk[axis] + self.hdr_I[axis]
            output[axis] = self.hdr_out[1, axis]
            
            # 가중치 계산
            if abs(pos_rtk[axis]) < self.hdr_thres[axis] or abs(acc[axis]) < self.hdr_thres[axis]:
                hdr_W = 1.0
            else:
                hdr_W = 0.0
            
            # 반복 계수 계산
            if self.get_sign(self.hdr_out[1, axis]) == self.get_sign(self.hdr_out[0, axis]):
                self.hdr_r[axis] += 1.0
                self.hdr_R[axis] = (1.0 + self.hdr_c1[axis]) / (1.0 + self.hdr_c1[axis] * np.power(self.hdr_r[axis], self.hdr_c2[axis]))
            else:
                self.hdr_r[axis] = 1.0
                self.hdr_R[axis] = 1.0
            
            # 적분 보정
            self.hdr_I[axis] = self.hdr_I[axis] - hdr_W * self.hdr_R[axis] * self.get_sign(self.hdr_out[1, axis]) * self.hdr_ic
        
        return output


class KalmanFilter2Stage:
    """2단계 칼만 필터 - Edata.cpp 구현"""
    
    def __init__(self):
        # 상태 전이 행렬
        self.Ad = np.array([
            [1, DT, 0.5*DT*DT],
            [0, 1, DT],
            [0, 0, 1]
        ])
        
        self.Gd = np.array([[0.5*DT*DT], [DT], [0]])
        
        # 측정 행렬
        self.Hd3 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        self.Hd2 = np.array([
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        self.Cd3 = np.array([
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        
        self.Cd2 = np.array([
            [1, 0],
            [0, 1]
        ])
        
        # 노이즈 공분산
        self.rda = np.array([1.0, 1.0, 1.0])
        self.rdd = np.array([12.0, 12.0, 12.0])
        self.rdv = np.array([152500.0, 152500.0, 152500.0])
        
        # R3, R2 행렬 초기화
        self.R3 = []
        self.R2 = []
        for i in range(3):
            R3_i = np.diag([self.rdd[i], self.rdv[i], self.rda[i]])
            R2_i = np.diag([self.rdv[i], self.rda[i]])
            self.R3.append(R3_i)
            self.R2.append(R2_i)
        
        # 상태 변수 초기화
        self.xBF = [np.zeros((3, 1)) for _ in range(3)]
        self.PBF = [np.eye(3) for _ in range(3)]
        self.KBF = [np.zeros((3, 3)) for _ in range(3)]
        
        self.bBE = [np.zeros((2, 1)) for _ in range(3)]
        self.PBE = [np.eye(2) for _ in range(3)]
        self.KBE = [np.zeros((2, 3)) for _ in range(3)]
        
        self.U = [np.zeros((3, 2)) for _ in range(3)]
        self.V = [np.zeros((3, 2)) for _ in range(3)]
        self.S = [np.zeros((3, 2)) for _ in range(3)]
        self.r = [np.zeros((3, 1)) for _ in range(3)]
        
        self.xSE = [np.zeros((3, 1)) for _ in range(3)]
        
        # 예측 속도
        self.pVel = np.zeros(3)
    
    def inv_matrix(self, mat):
        """행렬 역행렬 계산"""
        return np.linalg.inv(mat)
    
    def estimate(self, acc_data, rtk_data, fixed):
        """
        2단계 칼만 필터 추정
        Edata.cpp TwoStgKalmanAVD() 함수 참조 (87-148라인)
        """
        results = np.zeros((3, 3))  # [변위, 속도, 가속도] x 3축
        
        for i in range(3):
            # 측정값 구성
            if fixed > 1:  # GPS 고정 상태
                y = np.array([[rtk_data[i]], [self.pVel[i]], [acc_data[i]]])
                Rd = self.R3[i]
                Hd = self.Hd3
                Cd = self.Cd3
            else:  # GPS 미고정 상태
                y = np.array([[self.pVel[i]], [acc_data[i]]])
                Rd = self.R2[i]
                Hd = self.Hd2
                Cd = self.Cd2
            
            # PRIOR ESTIMATION
            # Bias-free 추정자
            self.xBF[i] = self.Ad @ self.xBF[i]
            self.PBF[i] = self.Ad @ self.PBF[i] @ self.Ad.T + self.Gd @ self.Gd.T
            
            # 감도 행렬 계산
            self.U[i] = self.Ad @ self.V[i]
            
            # POSTERIOR ESTIMATION
            # 칼만 이득 계산
            KBF_temp = self.PBF[i] @ Hd.T @ self.inv_matrix(Hd @ self.PBF[i] @ Hd.T + Rd)
            self.KBF[i] = KBF_temp
            
            # 감도 추정자
            self.S[i] = Hd @ self.U[i] + Cd
            self.r[i] = y - Hd @ self.xBF[i]
            self.V[i] = self.U[i] - self.KBF[i] @ self.S[i]
            
            # 편향 추정자
            P_temp = (Hd @ self.PBF[i] @ Hd.T + self.S[i] @ self.PBE[i] @ self.S[i].T + Rd)
            KBE_temp = self.PBE[i] @ self.S[i].T @ self.inv_matrix(P_temp)
            self.KBE[i] = KBE_temp
            
            self.bBE[i] = self.bBE[i] + self.KBE[i] @ (self.r[i] - self.S[i] @ self.bBE[i])
            self.PBE[i] = self.PBE[i] - self.KBE[i] @ self.S[i] @ self.PBE[i]
            
            # Bias-free 추정자 업데이트
            self.xBF[i] = self.xBF[i] + self.KBF[i] @ (y - Hd @ self.xBF[i])
            self.PBF[i] = self.PBF[i] - self.KBF[i] @ Hd @ self.PBF[i]
            
            # 최종 추정 (Sensitive Estimator)
            self.xSE[i] = self.xBF[i] + self.V[i] @ self.bBE[i]
            
            results[i, 0] = self.xSE[i][0]  # 변위
            results[i, 1] = self.xSE[i][1]  # 속도
            results[i, 2] = acc_data[i]     # 가속도
        
        return results


class GNSSDataProcessor:
    """GNSS 데이터 처리 메인 클래스"""
    
    def __init__(self):
        self.butterworth = ButterworthFilter()
        self.hdr = HDRProcessor()
        self.kalman = KalmanFilter2Stage()
    
    def process_csv(self, input_file, output_file, max_lines=100):
        """
        CSV 파일 처리 - Edata.cpp estimateDisp() 함수 참조 (165-259라인)
        
        입력 형식: Time, RTK_X, RTK_Y, RTK_Z, Acc_X, Acc_Y, Acc_Z, Fix, NumSat, Precision
        출력 형식: Time, ACC_X, ACC_Y, ACC_Z, VEL_X, VEL_Y, VEL_Z, DIS_X, DIS_Y, DIS_Z
        """
        print(f"입력 파일: {input_file}")
        print(f"출력 파일: {output_file}")
        
        # CSV 파일 읽기
        df = pd.read_csv(input_file)
        
        if len(df) > max_lines:
            df = df.iloc[:max_lines]
        
        # 결과 저장용 리스트
        results = []
        
        print(f"처리 중... (총 {len(df)} 행)")
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"진행률: {idx}/{len(df)}")
            
            # 데이터 추출
            time = row.iloc[0]  # 첫 번째 컬럼이 Time
            
            rtk_x = row['RTK_X'] if 'RTK_X' in row else row.iloc[1]
            rtk_y = row['RTK_Y'] if 'RTK_Y' in row else row.iloc[2]
            rtk_z = row['RTK_Z'] if 'RTK_Z' in row else row.iloc[3]
            
            acc_x = row['Acc_X'] if 'Acc_X' in row else row.iloc[4]
            acc_y = row['Acc_Y'] if 'Acc_Y' in row else row.iloc[5]
            acc_z = row['Acc_Z'] if 'Acc_Z' in row else row.iloc[6]
            
            fix = row['Fix'] if 'Fix' in row else row.iloc[7]
            num_sat = row['NumSat'] if 'NumSat' in row else row.iloc[8]
            prec = row['Precision'] if 'Precision' in row else row.iloc[9]
            
            # 가속도 조정 (Mdata.cpp 46라인: sensitivity = -10)
            sensitivity = -10.0
            acc_x *= sensitivity
            acc_y *= sensitivity
            acc_z *= sensitivity
            
            # Butterworth 저역 통과 필터 적용
            filtered_acc = self.butterworth.butterworth_lowpass_acc(acc_x, acc_y, acc_z)
            
            # GPS 데이터 처리
            pos_rtk = np.array([rtk_x, rtk_y, rtk_z])
            
            # HDR 처리 (GPS 고정 시)
            if fix > 1:
                pos_rtk = self.hdr.process(pos_rtk, filtered_acc)
            
            # 칼만 필터 추정
            kalman_results = self.kalman.estimate(filtered_acc, pos_rtk, fix)
            
            # 결과 저장
            result = {
                'Time': time,
                'ACC_X': kalman_results[0, 2],
                'ACC_Y': kalman_results[1, 2],
                'ACC_Z': kalman_results[2, 2],
                'VEL_X': kalman_results[0, 1],
                'VEL_Y': kalman_results[1, 1],
                'VEL_Z': kalman_results[2, 1],
                'DIS_X': kalman_results[0, 0],
                'DIS_Y': kalman_results[1, 0],
                'DIS_Z': kalman_results[2, 0]
            }
            
            results.append(result)
        
        # 결과를 DataFrame으로 변환
        output_df = pd.DataFrame(results)
        
        # CSV 파일로 저장
        output_df.to_csv(output_file, index=False)
        
        print(f"처리 완료! 결과: {output_file}")
        
        return output_df


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GNSS 변위 추정 시스템')
    parser.add_argument('input_file', help='입력 CSV 파일 경로')
    parser.add_argument('output_file', nargs='?', help='출력 CSV 파일 경로 (선택사항)')
    parser.add_argument('--max-lines', type=int, default=100, help='최대 처리 라인 수')
    
    args = parser.parse_args()
    
    # 출력 파일명 설정
    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        output_dir = os.path.dirname(args.input_file) or '.'
        args.output_file = os.path.join(output_dir, f'output_kalman-{base_name[15:]}' if len(base_name) > 15 else f'output_kalman-{base_name}')
    
    # 프로세서 생성 및 실행
    processor = GNSSDataProcessor()
    processor.process_csv(args.input_file, args.output_file, args.max_lines)


if __name__ == '__main__':
    main()
