# 차량 번호판 인식 프로젝트

이 프로젝트는 **EasyOCR** 라이브러리를 사용하여 차량 이미지에서 번호판을 인식하고, 해당 번호판을 출력하는 기능을 제공합니다. 번호판을 인식한 후 이미지를 출력하고, 번호판 텍스트를 추출합니다.

## 요구 사항

- Python 3.x
- OpenCV
- EasyOCR
- Matplotlib

## 설치

이 프로젝트를 실행하기 전에 필요한 라이브러리들을 설치하세요. 아래 명령어를 통해 필요한 패키지를 설치할 수 있습니다.

```bash
pip install easyocr opencv-python matplotlib
코드 설명
1. 라이브러리 임포트

import easyocr
import cv2
import matplotlib.pyplot as plt
easyocr: 이미지에서 텍스트를 인식하는 라이브러리
cv2: 이미지 처리 및 표시를 위한 OpenCV
matplotlib.pyplot: 이미지를 화면에 출력하기 위한 라이브러리
2. OCR 리더 객체 생성

reader = easyocr.Reader(['ko', 'en'], gpu=False)
easyocr.Reader를 사용하여 한글(ko)과 영어(en) 텍스트를 인식할 수 있는 리더 객체를 생성합니다.
GPU를 사용하지 않고 CPU에서만 동작하도록 설정합니다.
3. 이미지 읽기 및 출력

img_path = 'C:/workspace/mini_project/data/222.jpg'
img = cv2.imread(img_path)
이미지 경로를 지정하고 cv2.imread()를 통해 이미지를 읽습니다.
이미지가 정상적으로 열렸는지 확인하고, Matplotlib을 사용하여 이미지를 화면에 출력합니다.
4. OCR 실행 및 결과 처리

result = reader.readtext(img)
EasyOCR을 사용하여 이미지에서 텍스트를 추출합니다.
5. 번호판 인식 및 필터링

THRESHOLD = 0.1
plates = []
current_plate = ""

for bbox, text, conf in result:
    if conf > THRESHOLD:
        current_plate += text
        cv2.rectangle(img, pt1=tuple(map(int, bbox[0])), pt2=tuple(map(int, bbox[2])), color=(0, 255, 0), thickness=3)
        if len(current_plate) > 5:
            plates.append(current_plate)
            current_plate = ""
임계값(0.1)을 설정하여 신뢰도가 낮은 결과는 무시하고, 텍스트를 추출하여 번호판을 구성합니다.
번호판 텍스트가 일정 길이를 넘으면 새로운 번호판으로 분리하여 리스트에 저장합니다.
6. 번호판 출력

for idx, plate in enumerate(plates):
    print(f"차량 {idx+1} 번호판: {plate}")
추출된 번호판을 출력합니다.
7. 번호판 영역을 이미지에 표시

plt.figure(figsize=(8, 8))
plt.imshow(img[:, :, ::-1])
plt.axis('off')
plt.show()
인식된 번호판 영역을 표시한 이미지를 화면에 출력합니다.
사용 방법
프로젝트 디렉토리에 이미지 파일을 준비하고, 이미지 경로를 코드에서 img_path에 맞게 수정합니다.
코드를 실행하여 차량 번호판을 인식하고, 결과를 출력합니다.
결과 예시
출력 예시:

복사
편집
차량 1 번호판: 12가1234
차량 2 번호판: 34나5678
이미지: 번호판이 녹색 직사각형으로 표시된 이미지가 출력됩니다.

go
복사
편집

이렇게 구성된 `README.md` 파일을 사용하면 프로젝트의 목적과 코드 흐름을 잘 이해할 수 있고, 코드 실행 방법도 명확히 알 수 
