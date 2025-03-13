import torch
import easyocr
import cv2
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# YOLOv5 모델 로딩 (사전 훈련된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' 모델 사용 (빠르지만 정확도가 낮을 수 있음)

# EasyOCR 모델 로딩 (GPU 대신 CPU 사용)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# 동영상 파일 열기
video_path = 'C:/workspace/mini_project/data/test.mov'  # 동영상 파일 경로 지정
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
else:
    # 동영상 저장을 위한 VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 코덱 사용
    output_path = 'output_video.avi'  # 출력 동영상 파일 경로
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))  # 프레임 크기 및 FPS 설정

    while True:
        # 동영상에서 프레임을 읽어옴
        ret, img = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # YOLOv5로 모든 객체 감지
        results = model(img)

        # 감지된 객체의 좌표와 라벨 추출 (모든 객체 감지)
        car_detected = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.4:  # 신뢰도 기준을 낮추어 모든 객체를 감지하도록 변경
                x1, y1, x2, y2 = map(int, xyxy)
                car_detected.append((x1, y1, x2, y2))
                # 객체 영역 표시 및 라벨 추가
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 객체 영역 표시
                cv2.putText(img, f'{results.names[int(cls)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # 번호판 인식
        THRESHOLD = 0.1  # 임계값 설정
        plates = []
        current_plate = ""

        for bbox, text, conf in reader.readtext(img):  # EasyOCR을 통해 텍스트 읽기
            if conf > THRESHOLD:
                # 텍스트가 번호판 영역에 있을 경우만 처리
                for (x1, y1, x2, y2) in car_detected:
                    if x1 < bbox[0][0] < x2 and y1 < bbox[0][1] < y2:
                        current_plate += text  # 텍스트를 이어붙임
                        cv2.rectangle(img, pt1=tuple(map(int, bbox[0])), pt2=tuple(map(int, bbox[2])), color=(0, 255, 0), thickness=3)

                        # 번호판의 끝을 판단할 기준
                        if len(current_plate) > 5:  # 예시 기준: 번호판 길이가 5 이상이면 분리
                            plates.append(current_plate)
                            current_plate = ""

        # 마지막 번호판 추가
        if current_plate:
            plates.append(current_plate)

        # 차량 번호판 출력
        for idx, plate in enumerate(plates):
            print(f"차량 {idx+1} 번호판: {plate}")

        # 이미지 크기 조정 (비율 유지)
        height, width = img.shape[:2]
        max_size = 800  # 최대 크기 설정

        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = cv2.resize(img, (new_width, new_height))
        else:
            img_resized = img

        # OpenCV로 이미지 표시
        cv2.imshow("Detected Image", img_resized)

        # 동영상 파일에 프레임 저장
        out.write(img_resized)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 동영상 리소스 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
