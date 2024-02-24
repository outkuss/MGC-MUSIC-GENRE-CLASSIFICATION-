import os
import librosa
import numpy as np

# Blues klasöründeki WAV dosyalarının listesini al
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, "MGC", "archive", "data", "genres_original", "blues")

# Özelliklerin saklanacağı liste
features = []

# Her bir WAV dosyası için özellikleri çıkar
for wav_file in wav_files:
    y, sr = librosa.load(wav_file)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    melodi, _ = librosa.piptrack(y=y, sr=sr)  # Melodi özelliğini çıkar
    ritim = librosa.feature.tempogram(y=y, sr=sr)  # Ritim özelliğini çıkar
    spektral = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # Spektral özelliğini çıkar
    zaman = np.mean(librosa.feature.rms(y=y))  # Zaman özelliğini çıkar
    ton = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))  # Ton özelliğini çıkar
    
    # Özellikleri listeye ekle
    features.append([tempo, melodi, ritim, spektral, zaman, ton, 'blues'])

# Features listesini numpy array'e dönüştür
features = np.array(features)

# Veri setini ARFF formatına dönüştürerek kaydet
with open('music_genre.arff', 'w') as f:
    f.write('@relation music_genre\n')
    f.write('@attribute tempo numeric\n')
    f.write('@attribute melodi numeric\n')
    f.write('@attribute ritim numeric\n')
    f.write('@attribute spektral numeric\n')
    f.write('@attribute zaman numeric\n')
    f.write('@attribute ton numeric\n')
    f.write('@attribute genre {blues}\n')
    f.write('@data\n')
    for feature in features:
        f.write(','.join(map(str, feature)) + '\n')

print("Veri seti oluşturuldu ve music_genre.arff olarak kaydedildi")
