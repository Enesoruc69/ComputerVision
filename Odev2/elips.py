import cv2
import numpy as np
import matplotlib.pyplot as plt

# Elips fotoğrafı oluşturma
height, width = 200, 300  # Elipsin boyutları
img = np.zeros((height, width, 3), dtype=np.uint8)  # Siyah arka plan

# Elipsi oluştur
center = (width // 2, height // 2)  # Elipsin merkezi
axes = (100, 50)  # Yarım eksen boyutları (a, b)
angle = 0  # Elipsin açısı
start_angle = 0
end_angle = 360
cv2.ellipse(img, center, axes, angle, start_angle, end_angle, (255, 255, 255), -1)  # Beyaz elips

# Fotoğrafı gri tonlamaya çevirme
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Yatay türev filtresi [-1, 1]
horizontal_filter = np.array([[-1, 1]])

# Yatay türevi hesaplama
horizontal_derivative = cv2.filter2D(gray_img, -1, horizontal_filter)

# Dikey türev filtresi [-1, 1]T
vertical_filter = np.array([[-1], [1]])

# Dikey türevi hesaplama
vertical_derivative = cv2.filter2D(gray_img, -1, vertical_filter)

# Türevlerin min-max değerlerini kontrol etme
print("Horizontal Derivative Min-Max:", horizontal_derivative.min(), horizontal_derivative.max())
print("Vertical Derivative Min-Max:", vertical_derivative.min(), vertical_derivative.max())

# Türev büyüklüğünü hesaplama (horizontal ve vertical türevlerin birleşimi)
magnitude = np.sqrt(np.square(horizontal_derivative) + np.square(vertical_derivative))

# Büyüklük hesaplamasının sonucunu kontrol etme
print("Magnitude Min-Max:", magnitude.min(), magnitude.max())

# Eğer büyüklük küçükse, ölçekleme yapalım
if magnitude.max() <= 5:
    magnitude *= 50  # Küçük değerleri daha anlamlı hale getirmek için ölçekleme

# Kendi normalizasyon fonksiyonumuzu kullanalım
magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255

# Orijinal ve sonuç görüntülerini yan yana gösterme
plt.figure(figsize=(10, 5))

# Orijinal renkli fotoğraf
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Orijinal fotoğrafı BGR'den RGB'ye çevirdik
plt.title('Orijinal Fotoğraf')

# Yatay ve dikey türevlerin birleşimi (kenar tespiti)
plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Yatay ve Dikey Türevlerin Birleşimi')

plt.show()