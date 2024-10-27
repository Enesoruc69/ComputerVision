import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kare fotoğrafı yükleme
img = cv2.imread('kare.png')

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

# Yatay ve dikey türevlerin birleşimi (renk haritası ile)
plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='plasma')  # 'plasma' renk haritasını kullandık
plt.title('Yatay ve Dikey Türevlerin Birleşimi (Renkli)')

plt.show()