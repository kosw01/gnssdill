"""
GNSS 변위 추정 시스템 - Z축 전용 버전
Z축 변위만 추정하는 간소화된 버전
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Optional

# 상수 정의
DT = 0.01  # 샘플링 시간 간격
MAX_LINE = 100  # 최대 처리 라인 수


class ButterworthFilterZ:
    """Butterworth 필터 클래스 - Z축 전용"""
    
    def __init__(self):
        # 저역 통과 필터 계수 (Mdata.cpp 100-127라인 참조)
        self.gain = 779.6778047
        self.butter_low_acc_in = np.zeros(6)  # z축만
        self.butter_low_acc_out = np.zeros(6)
    
    def butterworth_lowpass_acc(self, acc_z):
        """
        가속도계 Z축 데이터에 Butterworth 저역 통과 필터 적용
        Mdata.cpp 100-127라인 참조
        """
        # 큐 시프트
        for i in range(5):
            self.butter_low_acc_in[i] = self.butter_low_acc_in[i+1]
            self.butter_low_acc_out[i] = self.butter_low_acc_out[i+1]
        
        # 새로운 입력 추가
        self.butter_low_acc_in[5] = acc_z / self.gain
        
        # FIR 필터 계수 적용 (IIR 형태)
        filtered_acc = (
            (self.butter_low_acc_in[0] + self.butter_low_acc_in[5]) +
            5 * (self.butter_low_acc_in[1] + self.butter_low_acc_in[4]) +
            10 * (self.butter_low_acc_in[2] + self.butter_low_acc_in[3]) +
            0.1254306222 * self.butter_low_acc_out[0] +
            -0.8811300754 * self.butter_low_acc_out[1] +
            2.5452528683 * self.butter_low_acc_out[2] +
            -3.8060181193 * self.butter_low_acc_out[3] +
            2.9754221097 * self.butter_low_acc_out[4]
        )
        self.butter_low_acc_out[5] = filtered_acc
        
        return filtered_acc


class HDRProcessorZ:
    """HDR (Heritage Dynamic Record) 처리 클래스 - Z축 전용"""
    
    def __init__(self):
        self.hdr_out = np.zeros(2)  # 이전/현재 출력
        self.hdr_I = 0.015  # 적분 초기값
        self.hdr_thres = 20.0
        self.hdr_c1 = 1.3
        self.hdr_c2 = 0.2
        self.hdr_r = 1.0
        self.hdr_R = 1.0
        self.hdr_ic = 0.015
    
    def get_sign(self, val):
        """부호 반환"""
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0
    
    def process(self, pos_rtk_z, acc_z):
        """
        HDR 처리 - Z축 전용
        Mdata.cpp 186-252라인 참조
        """
        # 출력 업데이트
        self.hdr_out[0] = self.hdr_out[1]
        self.hdr_out[1] = pos_rtk_z + self.hdr_I
        output = self.hdr_out[1]
        
        # 가중치 계산
        if abs(pos_rtk_z) < self.hdr_thres or abs(acc_z) < self.hdr_thres:
            hdr_W = 1.0
        else:
            hdr_W = 0.0
        
        # 반복 계수 계산
        if self.get_sign(self.hdr_out[1]) == self.get_sign(self.hdr_out[0]):
            self.hdr_r += 1.0
            self.hdr_R = (1.0 + self.hdr_c1) / (1.0 + self.hdr_c1 * np.power(self.hdr_r, self.hdr_c2))
        else:
            self.hdr_r = 1.0
            self.hdr_R = 1.0
        
        # 적분 보정
        self.hdr_I = self.hdr_I - hdr_W * self.hdr_R * self.get_sign(self.hdr_out[1]) * self.hdr_ic
        
        return output


class KalmanFilter2StageZ:
    """2단계 칼만 필터 - Z축 전용"""
    
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
        
        # 노이즈 공분산 (Z축 값 사용)
        self.rda = 1.0
        self.rdd = 12.0
        self.rdv = 152500.0
        
        # R3, R2 행렬 초기화
        self.R3 = np.diag([self.rdd, self.rdv, self.rda])
        self.R2 = np.diag([self.rdv, self.rda])
        
        # 상태 변수 초기화
        self.xBF = np.zeros((3, 1))
        self.PBF = np.eye(3)
        self.KBF = np.zeros((3, 3))
        
        self.bBE = np.zeros((2, 1))
        self.PBE = np.eye(2)
        self.KBE = np.zeros((2, 3))
        
        self.U = np.zeros((3, 2))
        self.V = np.zeros((3, 2))
        self.S = np.zeros((3, 2))
        self.r = np.zeros((3, 1))
        
        self.xSE = np.zeros((3, 1))
        
        # 예측 속도
        self.pVel = 0.0
    
    def inv_matrix(self, mat):
        """행렬 역행렬 계산"""
        return np.linalg.inv(mat)
    
    def estimate(self, acc_data_z, rtk_data_z, fixed):
        """
        2단계 칼만 필터 추정 - Z축 전용
        Edata.cpp TwoStgKalmanAVD() 함수 참조 (87-148라인)
        """
        # 측정값 구성
        if fixed > 1:  # GPS 고정 상태
            y = np.array([[rtk_data_z], [self.pVel], [acc_data_z]])
            Rd = self.R3
            Hd = self.Hd3
            Cd = self.Cd3
        else:  # GPS 미고정 상태
            y = np.array([[self.pVel], [acc_data_z]])
            Rd = self.R2
            Hd = self.Hd2
            Cd = self.Cd2
        
        # PRIOR ESTIMATION
        # Bias-free 추정자
        self.xBF = self.Ad @ self.xBF
        self.PBF = self.Ad @ self.PBF @ self.Ad.T + self.Gd @ self.Gd.T
        
        # 감도 행렬 계산
        self.U = self.Ad @ self.V
        
        # POSTERIOR ESTIMATION
        # 칼만 이득 계산
        KBF_temp = self.PBF @ Hd.T @ self.inv_matrix(Hd @ self.PBF @ Hd.T + Rd)
        self.KBF = KBF_temp
        
        # 감도 추정자
        self.S = Hd @ self.U + Cd
        self.r = y - Hd @ self.xBF
        self.V = self.U - self.KBF @ self.S
        
        # 편향 추정자
        P_temp = (Hd @ self.PBF @ Hd.T + self.S @ self.PBE @ self.S.T + Rd)
        KBE_temp = self.PBE @ self.S.T @ self.inv_matrix(P_temp)
        self.KBE = KBE_temp
        
        self.bBE = self.bBE + self.KBE @ (self.r - self.S @ self.bBE)
        self.PBE = self.PBE - self.KBE @ self.S @ self.PBE
        
        # Bias-free 추정자 업데이트
        self.xBF = self.xBF + self.KBF @ (y - Hd @ self.xBF)
        self.PBF = self.PBF - self.KBF @ Hd @ self.PBF
        
        # 최종 추정 (Sensitive Estimator)
        self.xSE = self.xBF + self.V @ self.bBE
        
        # 예측 속도 업데이트 (다음 반복을 위해)
        self.pVel = self.xSE[1, 0]
        
        # 결과 반환: [변위, 속도, 가속도]
        return {
            'displacement': self.xSE[0, 0],
            'velocity': self.xSE[1, 0],
            'acceleration': acc_data_z
        }


class GNSSDataProcessorZ:
    """GNSS 데이터 처리 메인 클래스 - Z축 전용"""
    
    def __init__(self):
        self.butterworth = ButterworthFilterZ()
        self.hdr = HDRProcessorZ()
        self.kalman = KalmanFilter2StageZ()
    
    def process_csv(self, input_file, output_file, max_lines=100):
        """
        CSV 파일 처리 - Z축 전용
        
        입력 형식: Time, RTK_Z, Acc_Z, Fix, NumSat, Precision
        출력 형식: Time, ACC_Z, VEL_Z, DIS_Z
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
            
            # 데이터 추출 (컬럼명 또는 인덱스로 접근)
            time = row.iloc[0] if 'Time' not in row else row['Time']
            
            rtk_z = row['RTK_Z'] if 'RTK_Z' in row else row.iloc[1]
            acc_z = row['Acc_Z'] if 'Acc_Z' in row else row.iloc[2]
            fix = row['Fix'] if 'Fix' in row else row.iloc[3]
            num_sat = row['NumSat'] if 'NumSat' in row else row.iloc[4]
            prec = row['Precision'] if 'Precision' in row else row.iloc[5]
            
            # 가속도 조정 (Mdata.cpp 46라인: sensitivity = -10)
            sensitivity = -10.0
            acc_z *= sensitivity
            
            # Butterworth 저역 통과 필터 적용
            filtered_acc_z = self.butterworth.butterworth_lowpass_acc(acc_z)
            
            # HDR 처리 (GPS 고정 시)
            if fix > 1:
                rtk_z = self.hdr.process(rtk_z, filtered_acc_z)
            
            # 칼만 필터 추정
            kalman_result = self.kalman.estimate(filtered_acc_z, rtk_z, fix)
            
            # 결과 저장
            result = {
                'Time': time,
                'ACC_Z': kalman_result['acceleration'],
                'VEL_Z': kalman_result['velocity'],
                'DIS_Z': kalman_result['displacement']
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
    
    parser = argparse.ArgumentParser(description='GNSS 변위 추정 시스템 - Z축 전용')
    parser.add_argument('input_file', help='입력 CSV 파일 경로')
    parser.add_argument('output_file', nargs='?', help='출력 CSV 파일 경로 (선택사항)')
    parser.add_argument('--max-lines', type=int, default=100, help='최대 처리 라인 수')
    
    args = parser.parse_args()
    
    # 출력 파일명 설정
    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        output_dir = os.path.dirname(args.input_file) or '.'
        name_without_ext = os.path.splitext(base_name)[0]
        args.output_file = os.path.join(output_dir, f'{name_without_ext}_z_output.csv')
    
    # 프로세서 생성 및 실행
    processor = GNSSDataProcessorZ()
    processor.process_csv(args.input_file, args.output_file, args.max_lines)


if __name__ == '__main__':
    main()

