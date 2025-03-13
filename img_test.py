import torch
import easyocr
import cv2

# YOLOv5 모델 로딩 (사전 훈련된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' 모델 사용 (빠르지만 정확도가 낮을 수 있음)

# EasyOCR 모델 로딩 (GPU 대신 CPU 사용)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

img_path = 'C:/workspace/mini_project/data/222.jpg'
img = cv2.imread(img_path)

if img is None:
    print("이미지를 열 수 없습니다. 경로를 확인해주세요.")
else:
    # YOLOv5로 차량 감지
    results = model(img)

    # 감지된 객체의 좌표와 라벨 추출 (차량 클래스는 COCO 데이터셋의 2번 클래스 (car))
    car_detected = []
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.4 and int(cls) == 2:  # 차량 클래스 (2번 클래스)
            x1, y1, x2, y2 = map(int, xyxy)
            car_detected.append((x1, y1, x2, y2))
            # 차량 영역 표시 및 라벨 추가
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 차량 영역 표시
            cv2.putText(img, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)  # 'car' 라벨 크기 늘리기

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
    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()  # 창 닫기
