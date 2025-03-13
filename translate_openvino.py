import torch
import easyocr
from easyocr import get_ocr_model

# EasyOCR에서 사용하는 모델 로드
reader = easyocr.Reader(['ko', 'en'])
ocr_model = get_ocr_model()  # EasyOCR 내부 함수로 모델 로드

# 더미 입력 생성 (모델의 입력 크기에 맞게 설정)
dummy_input = torch.randn(1, 3, 32, 320)  # EasyOCR의 경우 기본적으로 32x320 크기 이미지를 입력으로 받음

# 모델을 ONNX 형식으로 저장
onnx_path = 'easyocr_model.onnx'
torch.onnx.export(ocr_model, dummy_input, onnx_path, verbose=True)

print(f"모델이 {onnx_path}로 저장되었습니다.")
