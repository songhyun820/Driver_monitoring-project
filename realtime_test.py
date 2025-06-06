import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    # YOLO 모델 로드
    model = YOLO('runs/exp12/weights/best.pt')
    
    # 웹캠 초기화 (인덱스 0으로 변경)
    cap = cv2.VideoCapture(0)
    
    # 웹캠 연결 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 다른 인덱스를 시도해보세요.")
        return
    
    # FPS 계산을 위한 변수들
    prev_time = 0
    curr_time = 0
    
    # 클래스 이름 정의
    class_names = ['eye_closed', 'mouth_open', 'phone', 'cigar']
    
    print("실시간 테스트를 시작합니다. 종료하려면 'q'를 누르세요.")
    
    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
            
        # 그레이스케일로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 다시 BGR로 변환 (YOLO 모델이 컬러 이미지를 기대하므로)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # YOLO 모델로 예측
        results = model(frame)
        
        # 결과 시각화
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 신뢰도 점수
                conf = float(box.conf[0])
                
                # 클래스
                cls = int(box.cls[0])
                
                # 신뢰도가 0.5 이상인 경우에만 표시
                if conf > 0.5:
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 클래스 이름과 신뢰도 표시
                    label = f'{class_names[cls]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS 표시
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 표시
        cv2.imshow('Driver Monitoring', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main() 