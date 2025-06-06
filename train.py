import os
from ultralytics import YOLO

# OpenMP 라이브러리 중복 문제 해결 (필요시 사용)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # YOLOv8 nano 모델을 사전 학습된 가중치로 로드
    model = YOLO('yolov8n.pt') 
    
    # 모델 학습
    model.train(
        data='data/data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        workers=8,         # 추천: 0보다 큰 값으로 설정하여 학습 속도 향상
        device=0,
        
        # --- 기본값에서 변경을 원하는 하이퍼파라미터만 지정 ---
        lr0=0.01,          # 추천: YOLOv8 기본값으로 변경
        
        # augmentation 관련: 불필요한 증강만 비활성화
        degrees=0.0,       # 회전 증강 비활성화
        shear=0.0,         # 전단 증강 비활성화
        perspective=0.0,   # 원근 증강 비활성화
        
        project='runs',
        name='exp'
    )

if __name__ == '__main__':
    main()