# Nghiên cứu Thực nghiệm Ngưỡng CUTOFF của Mergesort

Dự án thực nghiệm khảo sát và xác định ngưỡng `CUTOFF` tối ưu khi kết hợp **Insertion Sort** vào **Merge Sort** cho các mảng con kích thước nhỏ theo giáo trình *Algorithms, 4th Edition* (Robert Sedgewick & Kevin Wayne).

---

## 1. Mục tiêu & Kết quả thực nghiệm chính

- **Mục tiêu:** Kiểm chứng tính đúng đắn, tính ổn định và đo lường sự cải thiện tốc độ của Mergesort khi áp dụng Insertion Sort ở các mảng con kích thước <= CUTOFF trên nhiều phân phối dữ liệu đầu vào.
- **Cấu hình máy thử nghiệm:** HP Pavilion 15-cs3xxx | Intel Core i5-1035G1 (4 nhân, 8 luồng) | 8 GB RAM | Windows 11 Pro 64-bit | OpenJDK 26.0.1 | Python 3.13.18.

### Kết quả CUTOFF tối ưu tóm tắt

| Phân phối dữ liệu | Kích thước N | CUTOFF tối ưu | Median tối ưu | Speedup vs C=1 | Vùng tối ưu thực tế (Wilcoxon p ≥ 0.05) |
| :--- | ---: | :---: | ---: | ---: | :---: |
| **`RANDOM`** | 1,250,000 | **6** | 566.33 ms | **+11.02%** | C ∈ [6, 10] (chuẩn hóa đa N: tối ưu là **7**) |
| **`PARTIALLY_SORTED`** | 1,250,000 | **128** | 137.91 ms | **+33.14%** | C ∈ [96, 128] (tận dụng O(N+I) của Insertion Sort) |
| **`REVERSED`** | 1,250,000 | **7** | 185.91 ms | **+10.61%** | C ∈ [6, 8] (C > 16 suy giảm do chi phí bậc hai O(C²)) |
| **`FEW_UNIQUE`** | 1,250,000 | **32** | 220.71 ms | **+17.30%** | C ∈ [16, 32] |

> Báo cáo chi tiết và kiểm định thống kê chuyên sâu xem tại [summary_report_tables.md](results/summary_report_tables.md) và [BaoCao_Mergesort_CUTOFF.pdf](reports/BaoCao_Mergesort_CUTOFF.pdf).

---

## 2. Cấu trúc thư mục & Tệp tin

### 2.1. Cây thư mục dự án
```text
PROJECT_Mergesort_CUTOFF_Improvement_25020431/
├── src/
│   ├── MergesortCutoff.java
│   ├── SortValidator.java
│   ├── DataFactory.java
│   └── ExperimentRunner.java
├── lib/
│   └── algs4.jar
├── bin/
├── datasets/
│   ├── dataset_manifest.csv
│   ├── random_N100000.txt
│   ├── partially_sorted_N100000.txt
│   ├── reversed_N100000.txt
│   └── few_unique_N100000.txt
├── results/
│   ├── environment_info.txt
│   ├── timing_raw.csv
│   ├── ops_count.csv
│   ├── doubling_test.csv
│   └── summary_report_tables.md
├── analysis/
│   └── analyze_results.py
├── figures/
│   ├── fig1_cutoff_curve_random.png
│   ├── fig2_speedup_by_distribution.png
│   ├── fig3_cost_model_vs_time.png
│   ├── fig4_doubling_loglog.png
│   └── fig5_variance_boxplot.png
└── reports/
    ├── BaoCao_Mergesort_CUTOFF.docx / .pdf
    └── Log chat với AI (GEMINI).docx / .pdf
```

### 2.2. Bảng vai trò từng thành phần

| Thư mục / Tệp tin | Vai trò chính |
| :--- | :--- |
| `src/MergesortCutoff.java` | Cài đặt thuật toán Mergesort cải tiến: chuyển sang Insertion Sort khi cỡ mảng con <= CUTOFF. |
| `src/SortValidator.java` | Kiểm thử tính đúng đắn và tính ổn định (Stability) trên 19 mốc CUTOFF với các ca biên và phân phối. |
| `src/DataFactory.java` | Bộ sinh dữ liệu tất định (deterministic seed), đếm nghịch thế O(N log N) và xuất file dữ liệu mẫu. |
| `src/ExperimentRunner.java` | Bộ điều phối thí nghiệm tự động: JIT Warm-up, quét thời gian thực, Doubling Test và đếm phép so sánh. |
| `lib/algs4.jar` | Thư viện thuật toán chuẩn của Princeton (cung cấp `Insertion`, `Stopwatch`, `In`). |
| `bin/` | Thư mục chứa bytecode Java (`.class`) đã biên dịch từ `src/`. |
| `datasets/` | Dữ liệu mẫu N = 100,000 cho 4 phân phối kèm bảng băm `dataset_manifest.csv` (SHA-256). |
| `results/` | Dữ liệu đo thô (`timing_raw.csv`, `ops_count.csv`, `doubling_test.csv`) và báo cáo số liệu (`summary_report_tables.md`). |
| `analysis/analyze_results.py` | Script Python tính toán thống kê (Bootstrap CI, Wilcoxon, Log-Log) và tự động vẽ 5 biểu đồ 300 DPI. |
| `figures/` | 5 biểu đồ trực quan hóa kết quả thực nghiệm (chuẩn 300 DPI, bảng màu Okabe-Ito). |
| `reports/` | Báo cáo hoàn chỉnh (`BaoCao_Mergesort_CUTOFF.docx/.pdf`) và nhật ký trao đổi AI (`Log chat với AI (GEMINI).docx/.pdf`). |

---

## 3. Hướng dẫn tái lập thực nghiệm (Reproducibility)

Thực thi tuần tự các câu lệnh sau trong **PowerShell** hoặc **CMD** tại thư mục gốc của dự án trên Windows:

### Bước 1: Cài đặt thư viện Python phục vụ vẽ biểu đồ
```powershell
pip install numpy pandas scipy matplotlib
```

### Bước 2: Biên dịch mã nguồn Java
```powershell
javac -encoding UTF-8 -cp "lib/algs4.jar" -d bin src/*.java
```

### Bước 3: Chạy kiểm thử tính đúng đắn & tính ổn định
```powershell
java -cp "bin;lib/algs4.jar" SortValidator
```

### Bước 4: Sinh lại bộ dữ liệu mẫu (Tùy chọn)
```powershell
java -cp "bin;lib/algs4.jar" DataFactory
```

### Bước 5: Chạy lại thí nghiệm đo đạc hiệu năng
- **Chạy toàn bộ thí nghiệm chuẩn (Khuyến nghị):**
  ```powershell
  java -Xms1536m -Xmx1536m -XX:+AlwaysPreTouch -cp "bin;lib/algs4.jar" ExperimentRunner
  ```
- **Hoặc chạy nhanh kiểm tra đường ống (Quick Mode < 1 phút):**
  ```powershell
  java -cp "bin;lib/algs4.jar" ExperimentRunner --quick
  ```

### Bước 6: Tái tạo toàn bộ biểu đồ & Bảng tổng hợp số liệu
```powershell
python analysis/analyze_results.py
```
> Script sẽ cập nhật toàn bộ biểu đồ trong thư mục `figures/` và xuất lại bảng số liệu tại `results/summary_report_tables.md`.
