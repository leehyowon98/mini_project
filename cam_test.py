## 수정해야됨


import sys
import torch
import easyocr
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # YOLOv5 모델 로드
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # EasyOCR 모델 로드
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)

        # 카메라 캡처
        self.cap = cv2.VideoCapture(0)  # 기본 카메라

        # 카메라 스트리밍을 위한 스레드 생성
        self.thread = VideoCaptureThread(self.cap)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # 타이머를 설정하여 실시간으로 프레임 갱신
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms마다 프레임 갱신

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템')
        self.setGeometry(100, 100, 800, 600)

        # 이미지 표시 라벨
        self.label_img = QLabel(self)
        self.label_img.setFixedSize(640, 480)

        # 번호판 리스트 출력
        self.list_plates = QListWidget(self)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.label_img)
        layout.addWidget(self.list_plates)
        self.setLayout(layout)

    def update_frame(self):
        """타이머로 갱신되는 프레임 처리"""
        if self.thread.latest_frame is not None:
            plates, processed_img = self.detectLicensePlates(self.thread.latest_frame)
            self.displayImage(processed_img)
            self.displayPlates(plates)

    def update_image(self, frame):
        """스레드에서 받은 프레임을 최신 프레임으로 업데이트"""
        self.thread.latest_frame = frame

    def detectLicensePlates(self, img):
        """번호판 인식 처리"""
        # YOLOv5로 차량 감지
        results = self.model(img)
        car_detected = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.4 and int(cls) == 2:  # 차량 클래스
                x1, y1, x2, y2 = map(int, xyxy)
                car_detected.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # EasyOCR을 이용한 번호판 인식
        THRESHOLD = 0.1
        plates = []
        current_plate = ""

        for bbox, text, conf in self.reader.readtext(img):
            if conf > THRESHOLD:
                for (x1, y1, x2, y2) in car_detected:
                    if x1 < bbox[0][0] < x2 and y1 < bbox[0][1] < y2:  # 차량 내부의 문자만 인식
                        current_plate += text
                        cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 255), 3)

                        if len(current_plate) > 5:
                            plates.append(current_plate)
                            current_plate = ""

        if current_plate:
            plates.append(current_plate)

        return plates, img

    def displayImage(self, img):
        """이미지 표시"""
        if img is None:
            return

        # OpenCV 이미지 → QImage 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # QLabel에 이미지 설정
        self.label_img.setPixmap(pixmap)
        self.label_img.setScaledContents(True)

    def displayPlates(self, plates):
        """번호판 리스트 출력"""
        self.list_plates.clear()
        for idx, plate in enumerate(plates):
            self.list_plates.addItem(f"차량 {idx + 1} 번호판: {plate}")

    def closeEvent(self, event):
        """윈도우 닫을 때 카메라 리소스 해제"""
        self.cap.release()
        self.thread.quit()
        event.accept()


class VideoCaptureThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.latest_frame = None

    def run(self):
        """비디오 캡처 및 프레임 송출"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.change_pixmap_signal.emit(frame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
