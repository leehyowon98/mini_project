차량 번호판 인식 시스템
이 프로젝트는 YOLOv5와 EasyOCR을 활용하여 차량 번호판을 실시간으로 인식하는 시스템입니다. PyQt5를 사용한 GUI로 실시간 영상 스트리밍과 번호판 인식 결과를 보여주며, 웹캠을 통해 차량 번호판을 캡처하고 인식합니다.

주요 기능
실시간 영상 스트리밍: 웹캠을 통해 실시간으로 영상을 스트리밍합니다.
차량 인식: YOLOv5 모델을 사용하여 차량을 인식합니다.
번호판 인식: EasyOCR을 사용하여 번호판을 인식하고 추출합니다.
GUI 화면: PyQt5를 사용하여 실시간 영상과 번호판 인식 결과를 화면에 표시합니다.
번호판 리스트: 인식된 번호판을 실시간으로 리스트에 출력합니다.
설치 및 실행 방법
1. 의존성 설치
이 프로젝트를 실행하려면 먼저 필요한 라이브러리를 설치해야 합니다. 아래 명령어를 사용하여 필요한 패키지를 설치하세요:

bash
복사
편집
pip install torch easyocr pyqt5 opencv-python numpy
2. 코드 실행
프로젝트를 실행하려면 아래 명령어를 사용하세요:

bash
복사
편집
python license_plate_recognition.py
실행 후, 웹캠을 통해 실시간으로 차량 번호판을 인식하며, 인식된 번호판은 GUI 화면에 표시됩니다. q 키를 눌러 프로그램을 종료할 수 있습니다.

3. 해상도 설정
웹캠의 해상도는 코드 내에서 기본적으로 1920x1080으로 설정되어 있습니다. 원하는 해상도로 변경하려면 아래 코드를 수정하세요:

python
복사
편집
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 해상도 설정 (너비)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 해상도 설정 (높이)
4. 프로그램 종료
실행 중 q 키를 눌러 프로그램을 종료할 수 있습니다.

코드 설명
LicensePlateRecognitionApp 클래스
PyQt5로 GUI를 구현한 클래스입니다.
실시간 영상 스트리밍과 번호판 인식 결과를 표시합니다.
VideoCaptureThread 클래스
별도의 스레드에서 실시간 영상 캡처를 처리하며, q 키를 누르면 캡처를 종료합니다.
detectLicensePlates 함수
YOLOv5 모델을 사용하여 차량을 감지하고, EasyOCR을 통해 번호판을 인식합니다.
displayImage 함수
OpenCV 이미지에서 QImage로 변환하여 GUI에 표시합니다.
displayPlates 함수
인식된 번호판을 리스트로 표시합니다.
프로젝트 구조
bash
복사
편집
license_plate_recognition.py   # 주요 코드 파일
README.md                     # 프로젝트 설명
참고 자료
YOLOv5
EasyOCR
PyQt5
OpenCV
