import os
import json
import shutil
from pathlib import Path

# --- 설정 (Configuration) ---
# 이 부분의 경로를 자신의 프로젝트 구조에 맞게 수정해주세요.
ANNOTATIONS_BASE_DIR = "data/data/annotations"  # 원본 JSON 파일들이 있는 기본 폴더
LABELS_OUTPUT_DIR = "data/Train/labels"      # 변환된 TXT 라벨 파일들을 저장할 폴더

# YOLO 클래스 매핑: 클래스 이름과 번호를 연결합니다.
CLASS_MAP = {
    'eye_closed': 0,
    'mouth_open': 1,
    'phone': 2,
    'cigar': 3
}

# --- 코드 (Code) ---

def convert_bbox_to_yolo(box, img_w, img_h):
    """
    [x_min, y_min, x_max, y_max] 형식의 바운딩 박스를
    YOLO 형식 (x_center, y_center, width, height)으로 변환합니다.
    모든 값은 0과 1 사이로 정규화됩니다.
    """
    x_min, y_min, x_max, y_max = box
    
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_json_to_yolo(json_path, output_path):
    """
    단일 JSON 파일을 읽고 분석하여 YOLO 형식의 라벨 파일을 생성합니다.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing {json_path}: {e}")
        return

    img_w = data['FileInfo']['Width']
    img_h = data['FileInfo']['Height']
    bbox_info = data['ObjectInfo']['BoundingBox']
    
    yolo_lines = []

    # ===== 1. 눈 감김 (eye_closed) - 수정된 로직 =====
    # 조건: 양쪽 눈이 모두 감겨있을 때
    if not bbox_info['Leye']['Opened'] and not bbox_info['Reye']['Opened']:
        # 왼쪽 눈이 보이면, 왼쪽 눈 영역을 'eye_closed'로 라벨링
        if bbox_info['Leye']['isVisible']:
            yolo_box_leye = convert_bbox_to_yolo(bbox_info['Leye']['Position'], img_w, img_h)
            yolo_lines.append(f"{CLASS_MAP['eye_closed']} {yolo_box_leye}")
        
        # 오른쪽 눈이 보이면, 오른쪽 눈 영역을 'eye_closed'로 라벨링
        if bbox_info['Reye']['isVisible']:
            yolo_box_reye = convert_bbox_to_yolo(bbox_info['Reye']['Position'], img_w, img_h)
            yolo_lines.append(f"{CLASS_MAP['eye_closed']} {yolo_box_reye}")

    # 2. 입 벌림 (mouth_open)
    if bbox_info['Mouth']['Opened'] and bbox_info['Mouth']['isVisible']:
        yolo_box = convert_bbox_to_yolo(bbox_info['Mouth']['Position'], img_w, img_h)
        yolo_lines.append(f"{CLASS_MAP['mouth_open']} {yolo_box}")

    # 3. 휴대폰 사용 (phone)
    if bbox_info['Phone']['isVisible']:
        yolo_box = convert_bbox_to_yolo(bbox_info['Phone']['Position'], img_w, img_h)
        yolo_lines.append(f"{CLASS_MAP['phone']} {yolo_box}")

    # 4. 흡연 (cigar)
    if bbox_info['Cigar']['isVisible']:
        yolo_box = convert_bbox_to_yolo(bbox_info['Cigar']['Position'], img_w, img_h)
        yolo_lines.append(f"{CLASS_MAP['cigar']} {yolo_box}")

    # 감지된 객체가 있든 없든 항상 라벨 파일을 생성 (빈 파일 포함)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

def main():
    """
    메인 실행 함수: 전체 데이터셋을 변환합니다.
    """
    if os.path.exists(LABELS_OUTPUT_DIR):
        shutil.rmtree(LABELS_OUTPUT_DIR)
    os.makedirs(LABELS_OUTPUT_DIR)
    print(f"'{LABELS_OUTPUT_DIR}' 폴더를 초기화했습니다.")

    json_files = list(Path(ANNOTATIONS_BASE_DIR).rglob("*.json"))
    
    if not json_files:
        print(f"'{ANNOTATIONS_BASE_DIR}'에서 JSON 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 {len(json_files)}개의 JSON 파일을 변환합니다...")

    for json_path in json_files:
        relative_path = json_path.relative_to(ANNOTATIONS_BASE_DIR)
        output_path = Path(LABELS_OUTPUT_DIR) / relative_path.with_suffix(".txt")
        
        process_json_to_yolo(json_path, output_path)

    print("YOLO 라벨 파일 변환 완료!")

if __name__ == "__main__":
    main()