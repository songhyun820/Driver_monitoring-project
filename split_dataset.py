import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir, train_ratio=0.8):
    """
    이미지와 라벨 폴더를 받아 학습 및 검증 세트로 분리합니다.

    - base_dir: 'images'와 'labels' 폴더를 포함하는 기본 데이터 폴더 경로
    - train_ratio: 학습 세트의 비율 (예: 0.8은 80%)
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'

    # 모든 이미지 파일 목록 가져오기 (하위 폴더 포함)
    all_images = sorted([p for p in images_dir.rglob('*.jpg')])
    
    # 데이터를 섞어서 무작위성을 보장
    random.seed(42)  # 항상 동일한 결과를 위해 시드 고정
    random.shuffle(all_images)

    # 학습 및 검증 세트 크기 계산
    split_point = int(len(all_images) * train_ratio)
    train_files = all_images[:split_point]
    val_files = all_images[split_point:]

    print(f"총 이미지 수: {len(all_images)}")
    print(f"학습용 이미지 수: {len(train_files)}")
    print(f"검증용 이미지 수: {len(val_files)}")

    # 새로운 폴더 구조 생성
    output_dir = base_dir.parent / 'split_data'
    train_img_out = output_dir / 'train/images'
    train_lbl_out = output_dir / 'train/labels'
    val_img_out = output_dir / 'val/images'
    val_lbl_out = output_dir / 'val/labels'
    
    # 기존 폴더가 있으면 삭제
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # 폴더 생성
    for p in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
        p.mkdir(parents=True, exist_ok=True)

    # 파일 복사 함수
    def copy_files(file_list, img_dest, lbl_dest):
        for img_path in file_list:
            # 라벨 파일 경로 생성 (e.g., .../images/001.jpg -> .../labels/001.txt)
            relative_img_path = img_path.relative_to(images_dir)
            lbl_path = labels_dir / relative_img_path.with_suffix('.txt')
            
            # 파일 복사
            shutil.copy(img_path, img_dest / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_dest / lbl_path.name)

    # 학습 및 검증 파일 복사 실행
    print("학습 파일 복사 중...")
    copy_files(train_files, train_img_out, train_lbl_out)
    print("검증 파일 복사 중...")
    copy_files(val_files, val_img_out, val_lbl_out)
    
    print("\n데이터 분리 완료!")
    print(f"새로운 데이터는 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == '__main__':
    # 'data/Train' 폴더 안에 images와 labels 폴더가 모두 있다고 가정
    # 이 경로는 자신의 폴더 구조에 맞게 수정해주세요.
    original_data_directory = 'data/train'
    split_dataset(original_data_directory, train_ratio=0.8)