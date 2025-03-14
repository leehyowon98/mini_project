# 차량 번호판 인식 시스템

이 프로젝트는 **YOLOv5**와 **EasyOCR**을 활용하여 차량 번호판을 실시간으로 인식하는 시스템입니다. **PyQt5**를 사용하여 GUI를 구성하고, 웹캠을 통해 실시간으로 번호판을 인식하여 화면에 출력합니다.

---

## 주요 기능

- **실시간 영상 스트리밍**  
  웹캠을 통해 실시간으로 영상을 스트리밍합니다.
  
- **차량 인식**  
  **YOLOv5** 모델을 사용하여 차량을 인식합니다.
  
- **번호판 인식**  
  **EasyOCR**을 사용하여 번호판을 인식하고 추출합니다.
  
- **GUI 화면**  
  **PyQt5**를 사용하여 실시간 영상과 번호판 인식 결과를 화면에 표시합니다.
  
- **번호판 리스트**  
  인식된 번호판을 실시간으로 리스트에 출력합니다.

---

## 프로젝트 구조
---

## 설치 및 실행 방법

### 1. 의존성 설치

이 프로젝트를 실행하기 위해 필요한 패키지를 먼저 설치해야 합니다. 아래 명령어를 사용하여 필요한 라이브러리를 설치하세요:

```bash
pip install torch easyocr pyqt5 opencv-python numpy
python license_plate_recognition.py
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 해상도 설정 (너비)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 해상도 설정 (높이)
---
