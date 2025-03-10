import os
import glob
import hashlib
import pickle

def get_file_hash(file_path):
    """파일의 MD5 해시를 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_file_hashes(directory, hash_file_path):
    """지정된 폴더의 모든 파일 해시를 계산하여 저장합니다."""
    # 대상 파일 검색 (여기서는 pdf, docx, pptx만 검색)
    files = glob.glob(os.path.join(directory, "*.pdf")) + \
            glob.glob(os.path.join(directory, "*.docx")) + \
            glob.glob(os.path.join(directory, "*.pptx"))

    # 해시 계산
    file_hashes = {file: get_file_hash(file) for file in files}

    # 해시 저장
    with open(hash_file_path, 'wb') as f:
        pickle.dump(file_hashes, f)
    
    print(f"파일 해시 저장 완료: {hash_file_path}")
    return file_hashes

# 테스트 실행
directory = "C:/Users/ASUS/test pdf"  # 해시를 계산할 폴더 경로
hash_file_path = "file_hashes.pkl"  # 해시 정보를 저장할 파일 경로

file_hashes = save_file_hashes(directory, hash_file_path)

# 결과 확인
print("저장된 파일 해시:")
for file, file_hash in file_hashes.items():
    print(f"{file}: {file_hash}")
