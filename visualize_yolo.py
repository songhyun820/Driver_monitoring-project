import cv2
import os
import glob
import numpy as np

def visualize_yolo_labels(image_path, label_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # 라벨 파일 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 클래스 이름 정의
    class_names = ['eye_closed', 'mouth_open', 'phone', 'cigar']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # BGR 형식
    
    # 각 라벨에 대해 바운딩 박스 그리기
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        
        # YOLO 형식을 픽셀 좌표로 변환
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # 바운딩 박스 그리기
        color = colors[int(class_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 클래스 이름 표시
        label = class_names[int(class_id)]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    # 이미지와 라벨 디렉토리 설정
    image_dir = 'data/train/images'  # 학습 이미지 디렉토리
    label_dir = 'data/train/labels'  # 학습 라벨 디렉토리
    
    # 하위 폴더까지 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
    
    if not image_files:
        print("이미지 파일을 찾을 수 없습니다.")
        return
    
    current_index = 0
    while True:
        # 현재 이미지 시각화
        image_path = image_files[current_index]
        # 하위 폴더 구조를 반영하여 라벨 경로 생성
        rel_path = os.path.relpath(image_path, image_dir)
        label_path = os.path.join(label_dir, os.path.splitext(rel_path)[0] + '.txt')
        
        if not os.path.exists(label_path):
            print(f"라벨 파일을 찾을 수 없습니다: {label_path}")
            return
        
        # 이미지 시각화
        visualized_image = visualize_yolo_labels(image_path, label_path)
        
        if visualized_image is not None:
            # 이미지 크기 조정 (선택사항)
            height, width = visualized_image.shape[:2]
            max_height = 800
            if height > max_height:
                scale = max_height / height
                width = int(width * scale)
                height = max_height
                visualized_image = cv2.resize(visualized_image, (width, height))
            
            # 결과 표시
            cv2.imshow('YOLO Labels Visualization', visualized_image)
            print(f"이미지: {image_path}\n라벨: {label_path}")
            print("다음 이미지: 'n', 이전 이미지: 'p', 종료: 'q'")
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_index = (current_index + 1) % len(image_files)
            elif key == ord('p'):
                current_index = (current_index - 1) % len(image_files)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 