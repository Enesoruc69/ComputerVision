import pandas as pd
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
import os
import re

# CSV'den linkleri oku
csv_file = "haberler.csv"
data = pd.read_csv(csv_file)

# YOLO modelini yükle
model = YOLO("yolo11n.pt")  # Pretrained YOLOv11n model

# Kayıt sonuç dosyası
results_file = "sonuclar.csv"

# Sonuçları tutmak için bir liste
sonuclar = []

# Her bir linkteki fotoğrafı al ve YOLO ile tespit yap
for idx, row in data.iterrows():
    url = row['link']  # CSV'deki link sütunu
    print(f"Processing {url}...")

    try:
        # Haber sayfasındaki fotoğrafı bul
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Sayfada resim URL'sini bul (örneğin, img tag'leri ile)
        img_tag = soup.find('img', src=re.compile(r'.*\.jpg'))
        if img_tag:
            img_url = img_tag['src']
            if not img_url.startswith("http"):  # Tam URL kontrolü
                img_url = requests.compat.urljoin(url, img_url)

            # Resmi indir
            img_data = requests.get(img_url).content
            img_filename = f"image_{idx}.jpg"
            with open(img_filename, 'wb') as handler:
                handler.write(img_data)
            print(f"Image saved as {img_filename}")

            # Resmi YOLO ile işle
            results = model(img_filename)

            # Sonuçlar boş mu kontrol et
            if not results:
                print(f"No results for {img_filename}")
                continue

            # En yüksek doğruluk oranına sahip nesneyi bul ve kaydet
            highest_conf = 0
            best_detection = None

            for result in results:
                boxes = result.boxes  # YOLO'nun bounding box çıktısı
                if boxes is None or len(boxes) == 0:
                    print(f"No objects detected in {img_filename}")
                    continue

                for box in boxes:
                    cls = int(box.cls.cpu().numpy()[0])  # tespit edilen sınıf
                    conf = box.conf.cpu().numpy()[0]  # tespit doğruluğu
                    class_name = model.names[cls]  # sınıf ismi

                    if conf > highest_conf:
                        highest_conf = conf
                        best_detection = {
                            'url': url,
                            'image': img_filename,
                            'class': class_name,
                            'confidence': conf
                        }

            # En yüksek doğruluk oranına sahip nesne sonucu detection_results'a eklenir
            if best_detection:
                sonuclar.append(best_detection)
                print(f"Best detection: {best_detection['class']} with confidence {best_detection['confidence']:.2f}")

        else:
            print(f"Image not found on {url}")

    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

# Sonuçları CSV dosyasına kaydet
if sonuclar:
    df_results = pd.DataFrame(sonuclar)
    df_results.to_csv(results_file, index=False)
    print("Detection results saved to CSV.")
else:
    print("No detection results to save.")
