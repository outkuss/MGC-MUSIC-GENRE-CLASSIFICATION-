% Dosya yolu ve dosya uzantısı
dosyaYolu = 'C:\Users\Utku Sina\OneDrive\Masaüstü\MGC\archive\Data\genres_original\rock\';
dosyaUzantisi = '.wav';

% Veri seti için boş matris oluşturma
veriSeti = [];

% Dosya adlarını işleme
dosyaListesi = dir([dosyaYolu, '*', dosyaUzantisi]); % Tüm .wav dosyalarını al

for i = 1:length(dosyaListesi)
    dosyaAdi = [dosyaYolu, dosyaListesi(i).name];
    
    % Dosyayı yükleme   
    [sinyal, fs] = audioread(dosyaAdi);
    
    % Sinyal analizi ve özellik çıkarımı
    enerji = sum(sinyal.^2);
    spektrum = abs(fft(sinyal));
    spektralMerkezFrekansi = sum(spektrum.*(1:length(spektrum))') / sum(spektrum);
    
    % Elde edilen özellikleri bir vektöre ekleme
    yeniSatir = [enerji, spektralMerkezFrekansi];
    veriSeti = [veriSeti; yeniSatir];
end

% Elde edilen veri setini kaydetme
save('rock_veriSeti.mat', 'veriSeti');


copyfile('rock_veriSeti.mat', 'indirmek istediğiniz dosya yolu')