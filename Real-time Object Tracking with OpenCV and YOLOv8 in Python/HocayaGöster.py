import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8n.pt')  # YOLOv8 model dosyanızı buraya ekleyin

# Kamerayı aç
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılırken bir hata oluştu.")
        break

    # Modeli kullanarak tahmin yap
    results = model(frame)

    # Sonuçları çerçeveye çiz
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Kutu koordinatları
            conf = box.conf[0].item()  # Güven skoru
            cls = int(box.cls[0].item())  # Sınıf
            label = f'Class: {cls}, Conf: {conf:.2f}'  # Etiket

            # Çerçeveye kutu ve etiket çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sonucu göster
    cv2.imshow("YOLOv8 Nesne Tanıma", frame)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
