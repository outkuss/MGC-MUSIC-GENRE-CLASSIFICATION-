import os
import numpy as np
from scipy.io import wavfile
from scipy.io import arff

# Dosya yolu ve dosya uzantısı
dosya_yolu = 'C:\\Users\\Utku Sina\\OneDrive\\Masaüstü\\MGC\\archive\\Data\\genres_original\\rock\\'
dosya_uzantisi = '.wav'

# Veri seti için boş liste oluşturma
veri_seti = []

# Dosya adlarını işleme
dosya_listesi = [dosya for dosya in os.listdir(dosya_yolu) if dosya.endswith(dosya_uzantisi)]

for dosya_adi in dosya_listesi:
    dosya_yolu_tam = os.path.join(dosya_yolu, dosya_adi)
    
    # Dosyayı yükleme
    fs, sinyal = wavfile.read(dosya_yolu_tam)
    
    # Sinyal analizi ve özellik çıkarımı
    enerji = np.sum(sinyal**2)
    spektrum = np.abs(np.fft.fft(sinyal))
    spektral_merkez_frekansi = np.sum(spektrum * np.arange(len(spektrum))) / np.sum(spektrum)
    
    # Elde edilen özellikleri bir vektöre ekleme
    yeni_satir = [enerji, spektral_merkez_frekansi]
    veri_seti.append(yeni_satir)

# Veri setini .arff formatında kaydetme
with open('rock_veriSeti.arff', 'w') as dosya:
    dosya.write('@relation music_genre\n')
    dosya.write('@attribute enerji numeric\n')
    dosya.write('@attribute spektral_merkez_frekansi numeric\n')
    dosya.write('@data\n')
    for veri in veri_seti:
        dosya.write(','.join(map(str, veri)) + '\n')
