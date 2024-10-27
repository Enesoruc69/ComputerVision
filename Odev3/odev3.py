import os
import cv2
from ultralytics import YOLO

# YOLO modelini yükleyin
model = YOLO("yolov8n.pt")  # Model dosyanızın yolunu belirtin

# Görüntülerin bulunduğu klasörün yolu
image_folder = "/Users/macbookpro/Desktop/resim_cv"  # Klasörün yolunu belirtin

# Sonuçların kaydedileceği yeni klasör
output_folder = "/Users/macbookpro/Desktop/sonuc_cv"  # Yeni klasör ismi
os.makedirs(output_folder, exist_ok=True)  # Klasör yoksa oluştur

# Klasördeki her görüntüyü işle
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Gerekli uzantıları kontrol edin
        image_path = os.path.join(image_folder, filename)

        # Görüntüyü okuyun
        image = cv2.imread(image_path)

        # Görüntüyü model ile işleyin
        results = model(image)

        # Sonuçları işleyin
        for result in results:
            for box in result.boxes:  # Her bir kutuyu işleyin
                class_id = int(box.cls[0])  # Sınıf kimliğini alın (ilk elemanı al)
                confidence = box.conf[0]  # Güven puanını alın (ilk elemanı al)
                label = model.names[class_id]  # Sınıf adını alın

                # Kutunun koordinatlarını alıp görüntü üzerine çizin
                x1, y1, x2, y2 = box.xyxy[0]  # İlk elemanı alarak dört değer al
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # İşlenmiş görüntüyü yeni klasöre kaydedin
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

        # İşlenmiş görüntüyü gösterin (isteğe bağlı)
        cv2.imshow("Result", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
