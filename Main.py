import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


# Between-Class Variance
def threshold_image(im, th):
    """Mengubah gambar menjadi hitam-putih (0 dan 255) berdasarkan nilai threshold."""
    thresholded_im = np.zeros(im.shape, dtype=np.uint8)
    thresholded_im[im >= th] = 255
    return thresholded_im

def compute_between_class_variance(im, th):
    """Menghitung kriteria Otsu (Between-Class Variance) untuk dimaksimalkan."""
    
    kelas0_pixels = im[im < th]
    kelas1_pixels = im[im >= th]
    
  
    if len(kelas0_pixels) == 0 or len(kelas1_pixels) == 0:
        return 0

    total_pixels = im.size
    w0 = len(kelas0_pixels) / total_pixels
    w1 = 1.0 - w0
    
    mu0 = np.mean(kelas0_pixels)
    mu1 = np.mean(kelas1_pixels)
    
    return w0 * w1 * ((mu0 - mu1) ** 2)

def jelaskan_perhitungan_otsu(im, t_manual):
    """Fungsi ini mencetak setiap langkah perhitungan Between-Class Variance."""
    print("="*60)
    print(f"ANALISIS PERHITUNGAN MATEMATIS UNTUK THRESHOLD t = {t_manual}")
    print("="*60)

    kelas0_pixels = im[im < t_manual]
    kelas1_pixels = im[im >= t_manual]
    
    if len(kelas0_pixels) == 0 or len(kelas1_pixels) == 0:
        print("\nSalah satu kelas kosong, perhitungan tidak dapat dilanjutkan.")
        return
        
    w0 = len(kelas0_pixels) / im.size
    w1 = len(kelas1_pixels) / im.size
    mu0 = np.mean(kelas0_pixels)
    mu1 = np.mean(kelas1_pixels)
    between_class_var = w0 * w1 * ((mu0 - mu1) ** 2)
    
    print(f"[LANGKAH 1: Hitung Bobot]")
    print(f"Bobot Kelas 0 (w0) = {w0:.4f}")
    print(f"Bobot Kelas 1 (w1) = {w1:.4f}")
    print(f"\n[LANGKAH 2: Hitung Rata-rata]")
    print(f"Rata-rata Kelas 0 (μ0) = {mu0:.4f}")
    print(f"Rata-rata Kelas 1 (μ1) = {mu1:.4f}")
    print(f"\n[LANGKAH 3: Hitung Between-Class Variance]")
    print(f"  σB^2 = w0 * w1 * (μ0 - μ1)^2")
    print(f"  σB^2 = {w0:.4f} * {w1:.4f} * ({mu0:.4f} - {mu1:.4f})^2 = {between_class_var:.4f}")
    print("="*60)


#Algoritma Pencarian

def find_threshold_bruteforce(im):
    """Mencari threshold optimal dengan mencoba semua kemungkinan (0-255)."""
    print("\nMemulai pencarian dengan Brute-Force (Maximize Between-Class Variance)...")
    start_time = time.time()
    
    all_variances = [compute_between_class_variance(im, t) for t in range(256)]
    best_threshold = np.argmax(all_variances)
    
    end_time = time.time()
    print(f"Pencarian Brute-Force Selesai dalam {end_time - start_time:.4f} detik.")
    return best_threshold, all_variances

def binary_to_int(binary_list):
    return int("".join(str(x) for x in binary_list), 2)

def create_initial_population(size, chromosome_length=8):
    return [np.random.randint(0, 2, chromosome_length).tolist() for _ in range(size)]

def calculate_fitness(im, chromosome):
    threshold = binary_to_int(chromosome)
    return compute_between_class_variance(im, threshold)

def find_threshold_ga(im, population_size=50, generations=30, mutation_rate=0.01, elite_size_ratio=0.1):
    """Fungsi utama Algoritma Genetika."""
    print("\nMemulai pencarian dengan Algoritma Genetika...")
    start_time = time.time()
    
    population = create_initial_population(population_size)
    best_chromosome_overall = None
    best_fitness_overall = -1

    for gen in range(generations):
        fitness_scores = [calculate_fitness(im, chromo) for chromo in population]

        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness_overall:
            best_fitness_overall = fitness_scores[current_best_idx]
            best_chromosome_overall = population[current_best_idx]
        
        elite_size = int(population_size * elite_size_ratio)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = [population[i] for i in sorted_indices[:elite_size]]

        while len(new_population) < population_size:
            parent1_idx, parent2_idx = np.random.choice(elite_size, 2, replace=False)
            parent1, parent2 = new_population[parent1_idx], new_population[parent2_idx]
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            for i in range(len(child)):
                if np.random.rand() < mutation_rate:
                    child[i] = 1 - child[i]
            new_population.append(child)
            
        population = new_population
        print(f"Generasi {gen+1}/{generations} | Threshold Terbaik Sementara: {binary_to_int(best_chromosome_overall)}", end='\r')
    
    print("\n" + "="*50)
    end_time = time.time()
    print(f"Pencarian GA Selesai dalam {end_time - start_time:.2f} detik.")
    
    return binary_to_int(best_chromosome_overall)


try:
    path_im = r'C:\Users\User\Downloads\20230125_181143.jpg' # path gambar
    im_gray = np.asarray(Image.open(path_im).convert('L'))
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan. Menggunakan gambar dummy.")
    im_gray = np.zeros((200, 200), dtype=np.uint8)
    im_gray[50:150, 50:100] = 80
    im_gray[50:150, 100:150] = 170


THRESHOLD_UNTUK_DIANALISIS = 120 
jelaskan_perhitungan_otsu(im_gray, THRESHOLD_UNTUK_DIANALISIS)

t_bruteforce, all_variances = find_threshold_bruteforce(im_gray)
t_ga = find_threshold_ga(im_gray)


print("\n" + "="*50)
print("HASIL PERBANDINGAN NILAI THRESHOLD")
print("="*50)
print(f"--> Threshold dari Brute-Force (Optimal) : {t_bruteforce}")
print(f"--> Threshold dari Algoritma Genetika    : {t_ga}")
selisih = abs(t_bruteforce - t_ga)
if selisih == 0:
    print("--> KESIMPULAN: Algoritma Genetika berhasil menemukan nilai optimal!")
else:
    print(f"--> KESIMPULAN: Algoritma Genetika mendekati nilai optimal dengan selisih {selisih}.")
print("="*50)


print("\nMenyiapkan plot perbandingan hasil akhir...")


fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Perbandingan Hasil dan Analisis Metode Otsu', fontsize=20)

#gray scale
axes[0, 0].imshow(im_gray, cmap='gray')
axes[0, 0].set_title('1. Gambar Grayscale (Input)')
axes[0, 0].axis('off')

#grafik otsu
ax1 = axes[0, 1]
ax1.set_title('2. Analisis Kurva Between-Class Variance')
ax1.set_xlabel('Intensitas Piksel (Threshold)')
ax1.set_ylabel('Jumlah Piksel (Histogram)', color='tab:blue')
ax1.hist(im_gray.ravel(), bins=256, color='tab:blue', alpha=0.6, label='Histogram')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Between-Class Variance (σB^2)', color=color)
ax2.plot(range(256), all_variances, color=color, label='Between-Class Variance')
ax2.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=t_bruteforce, color='green', linestyle='--', linewidth=2, label=f'Optimal (Brute-Force) = {t_bruteforce}')
ax1.axvline(x=t_ga, color='purple', linestyle=':', linewidth=3, label=f'Hasil GA = {t_ga}')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

# hasil ga
im_hasil_ga = threshold_image(im_gray, t_ga) 
axes[1, 0].imshow(im_hasil_ga, cmap='gray')
axes[1, 0].set_title(f'3. Hasil Algoritma Genetika (t = {t_ga})')
axes[1, 0].axis('off')

#hasil bruteforce
im_hasil_bruteforce = threshold_image(im_gray, t_bruteforce)
axes[1, 1].imshow(im_hasil_bruteforce, cmap='gray')
axes[1, 1].set_title(f'4. Hasil Brute-Force (Optimal) (t = {t_bruteforce})')
axes[1, 1].axis('off')

#gabungan plot
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()