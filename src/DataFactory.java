import edu.princeton.cs.algs4.In;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Bộ sinh và quản lý dữ liệu thí nghiệm chuẩn hóa cho dự án Mergesort CUTOFF.
 * - Đảm bảo tính tái lập tuyệt đối (deterministic seed theo phân phối và kích thước N).
 * - Hỗ trợ so sánh ghép cặp (paired comparison) qua cloneArray / copyInto.
 * - Xuất bộ dữ liệu nộp bài kèm file manifest kiểm chứng đặc trưng thực tế.
 */
public final class DataFactory {

    public static final long BASE_SEED = 20261120L;

    // Tham số kiểm soát phân phối PARTIALLY_SORTED:
    // Hoán đổi 10% * N cặp trong cửa sổ hẹp w = 8 => tổng nghịch thế <= (2w - 1) * 0.1N = 1.5N = Theta(N)
    public static final double PARTIAL_SWAP_RATIO = 0.10;
    public static final int PARTIAL_WINDOW = 8;

    // Tham số kiểm soát phân phối FEW_UNIQUE: 10 khóa phân biệt (0..9)
    public static final int FEW_UNIQUE_KEYS = 10;

    // Kích thước mặc định khi xuất bộ dữ liệu mẫu vào datasets/ (~2.0 MB tổng cộng cho 4 file)
    public static final int DEFAULT_EXPORT_N = 100_000;

    private DataFactory() {}

    public enum Distribution {
        RANDOM("random", "knuth_shuffle"),
        PARTIALLY_SORTED(
                "partially_sorted",
                String.format(Locale.US, "local_swaps=%.0f%%;window=%d", PARTIAL_SWAP_RATIO * 100, PARTIAL_WINDOW)
        ),
        REVERSED("reversed", "strictly_decreasing"),
        FEW_UNIQUE("few_unique", "distinct_pool=" + FEW_UNIQUE_KEYS);

        private final String slug;
        private final String paramsDescription;

        Distribution(String slug, String paramsDescription) {
            this.slug = slug;
            this.paramsDescription = paramsDescription;
        }

        public String slug() {
            return slug;
        }

        public String paramsDescription() {
            return paramsDescription;
        }
    }

    // ==================== SEED & GENERATION ====================

    /**
     * Công thức sinh seed tất định từ cặp (phân phối, kích thước N).
     */
    public static long computeSeed(Distribution dist, int n) {
        return BASE_SEED + (dist.ordinal() + 1L) * 1_000_003L + (long) n * 31L;
    }

    public static Integer[] generate(Distribution dist, int n) {
        return generate(dist, n, computeSeed(dist, n));
    }

    public static Integer[] generate(Distribution dist, int n, long seed) {
        if (n < 0) {
            throw new IllegalArgumentException("Kich thuoc N phai >= 0, nhan duoc: " + n);
        }
        return switch (dist) {
            case RANDOM -> generateRandom(n, seed);
            case PARTIALLY_SORTED -> generatePartiallySorted(n, seed, PARTIAL_SWAP_RATIO, PARTIAL_WINDOW);
            case REVERSED -> generateReversed(n);
            case FEW_UNIQUE -> generateFewUnique(n, seed, FEW_UNIQUE_KEYS);
        };
    }

    /**
     * RANDOM: Khởi tạo [0..N-1] và xáo trộn đều bằng Knuth Shuffle (Fisher-Yates).
     */
    public static Integer[] generateRandom(int n, long seed) {
        Integer[] a = generateAscending(n);
        Random rng = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int r = rng.nextInt(i + 1);
            Integer tmp = a[i];
            a[i] = a[r];
            a[r] = tmp;
        }
        return a;
    }

    /**
     * PARTIALLY_SORTED: Khởi tạo mảng tăng dần, thực hiện floor(swapRatio * N) phép đổi chỗ
     * trong cửa sổ hẹp [i+1, min(i+window, N-1)] để số cặp nghịch thế tuyến tính theo N.
     */
    public static Integer[] generatePartiallySorted(int n, long seed, double swapRatio, int window) {
        Integer[] a = generateAscending(n);
        if (n <= 1) return a;

        Random rng = new Random(seed);
        int swaps = Math.max(1, (int) Math.round(n * swapRatio));
        for (int s = 0; s < swaps; s++) {
            int i = rng.nextInt(n - 1);
            int maxStep = Math.min(window, n - 1 - i);
            int j = i + 1 + rng.nextInt(maxStep);
            Integer tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
        return a;
    }

    /**
     * REVERSED: Mảng giảm dần ngặt [N-1, N-2, ..., 0].
     */
    public static Integer[] generateReversed(int n) {
        Integer[] a = new Integer[n];
        for (int i = 0; i < n; i++) {
            a[i] = n - 1 - i;
        }
        return a;
    }

    /**
     * FEW_UNIQUE: Mảng chỉ gồm các khóa thuộc tập {0, 1, ..., numKeys - 1}.
     */
    public static Integer[] generateFewUnique(int n, long seed, int numKeys) {
        if (numKeys <= 0) {
            throw new IllegalArgumentException("So khoa phan biet phai > 0");
        }
        Integer[] a = new Integer[n];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            a[i] = rng.nextInt(numKeys);
        }
        return a;
    }

    private static Integer[] generateAscending(int n) {
        Integer[] a = new Integer[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        return a;
    }

    // ==================== CLONE / PAIRED COPY ====================

    /**
     * Tạo mảng bản sao độc lập (shallow copy tham chiếu Integer bất biến).
     */
    public static Integer[] cloneArray(Integer[] source) {
        return Arrays.copyOf(source, source.length);
    }

    /**
     * Reset mảng làm việc từ mảng gốc mà không cấp phát bộ nhớ mới (giảm rác cho GC khi benchmark).
     */
    public static void copyInto(Integer[] source, Integer[] target) {
        if (source.length != target.length) {
            throw new IllegalArgumentException("Do dai mang nguon va mang dich khong khop");
        }
        System.arraycopy(source, 0, target, 0, source.length);
    }

    // ==================== FILE I/O ====================

    /**
     * Xuất mảng ra file .txt (mỗi dòng 1 số nguyên, chuẩn UTF-8).
     */
    public static void writeToFile(Integer[] data, Path filePath) throws IOException {
        Path parent = filePath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(filePath, StandardCharsets.UTF_8)) {
            for (Integer val : data) {
                writer.write(val.toString());
                writer.newLine();
            }
        }
    }

    /**
     * Đọc nhanh bộ dữ liệu từ file .txt bằng BufferedReader.
     */
    public static Integer[] loadFromFile(Path filePath) throws IOException {
        List<Integer> list = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(filePath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) {
                    list.add(Integer.valueOf(trimmed));
                }
            }
        }
        return list.toArray(new Integer[0]);
    }

    /**
     * Đọc bộ dữ liệu bằng lớp In của thư viện algs4.jar (dùng đối chiếu khi cần).
     */
    public static Integer[] loadFromFileAlgs4(String filePath) {
        In in = new In(filePath);
        int[] raw = in.readAllInts();
        Integer[] result = new Integer[raw.length];
        for (int i = 0; i < raw.length; i++) {
            result[i] = raw[i];
        }
        return result;
    }

    // ==================== DATASET VERIFICATION METRICS ====================

    public static int countDistinctKeys(Integer[] a) {
        if (a.length == 0) return 0;
        int[] copy = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            copy[i] = a[i];
        }
        Arrays.sort(copy);
        int distinct = 1;
        for (int i = 1; i < copy.length; i++) {
            if (copy[i] != copy[i - 1]) {
                distinct++;
            }
        }
        return distinct;
    }

    /**
     * Đếm chính xác số cặp nghịch thế (i < j mà a[i] > a[j]) trong thời gian O(N log N)
     * bằng kỹ thuật nén tọa độ + Fenwick Tree (Binary Indexed Tree), độc lập với Mergesort.
     */
    public static long countInversions(Integer[] a) {
        int n = a.length;
        if (n <= 1) return 0L;

        int[] unique = new int[n];
        for (int i = 0; i < n; i++) {
            unique[i] = a[i];
        }
        Arrays.sort(unique);
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (i == 0 || unique[i] != unique[i - 1]) {
                unique[m++] = unique[i];
            }
        }

        int[] bit = new int[m + 1];
        long inversions = 0L;
        for (int i = n - 1; i >= 0; i--) {
            int rank = Arrays.binarySearch(unique, 0, m, a[i]) + 1; // 1-indexed
            inversions += bitQuery(bit, rank - 1);
            bitUpdate(bit, rank, 1);
        }
        return inversions;
    }

    private static void bitUpdate(int[] bit, int idx, int delta) {
        for (int i = idx; i < bit.length; i += i & -i) {
            bit[i] += delta;
        }
    }

    private static int bitQuery(int[] bit, int idx) {
        int sum = 0;
        for (int i = idx; i > 0; i -= i & -i) {
            sum += bit[i];
        }
        return sum;
    }

    private static String sha256File(Path filePath) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream is = Files.newInputStream(filePath)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    md.update(buffer, 0, read);
                }
            }
            byte[] digest = md.digest();
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("Khong tim thay thuat toan SHA-256", e);
        }
    }

    // ==================== CLI EXPORT TOOL ====================

    /**
     * Xuất bộ dữ liệu đại diện đủ 4 phân phối vào thư mục datasets/ kèm dataset_manifest.csv.
     * Cách dùng:
     *   java -cp "bin;lib/algs4.jar" DataFactory            (mặc định N = 100000)
     *   java -cp "bin;lib/algs4.jar" DataFactory 200000     (tùy chọn N)
     */
    public static void main(String[] args) throws IOException {
        int exportN = (args.length > 0) ? Integer.parseInt(args[0]) : DEFAULT_EXPORT_N;
        Path outDir = Paths.get("datasets");
        Files.createDirectories(outDir);

        Path manifestPath = outDir.resolve("dataset_manifest.csv");
        long maxInversions = (long) exportN * (exportN - 1) / 2L;

        System.out.println("=== XUAT BO DU LIEU THI NGHIEM VAO THU MUC datasets/ ===");
        System.out.printf(Locale.US, "Kich thuoc chuan N = %,d | Max inversions = %,d%n", exportN, maxInversions);

        try (BufferedWriter manifest = Files.newBufferedWriter(manifestPath, StandardCharsets.UTF_8)) {
            manifest.write("filename,distribution,n,seed,params,distinct_keys,inversions,inversions_per_n,inversion_ratio,file_size_bytes,sha256");
            manifest.newLine();

            for (Distribution dist : Distribution.values()) {
                long seed = computeSeed(dist, exportN);
                Integer[] data = generate(dist, exportN, seed);

                String fileName = dist.slug() + "_N" + exportN + ".txt";
                Path dataFile = outDir.resolve(fileName);
                writeToFile(data, dataFile);

                // Kiểm chứng đọc ngược (round-trip) để đảm bảo file xuất ra vẹn toàn 100%
                Integer[] reloaded = loadFromFile(dataFile);
                if (!Arrays.equals(data, reloaded)) {
                    throw new IllegalStateException("Loi kiem chung doc nguoc file: " + fileName);
                }

                int distinctKeys = countDistinctKeys(data);
                long inversions = countInversions(data);
                double invPerN = (exportN > 0) ? (double) inversions / exportN : 0.0;
                double invRatio = (maxInversions > 0) ? (double) inversions / maxInversions : 0.0;
                long fileSize = Files.size(dataFile);
                String sha256 = sha256File(dataFile);

                manifest.write(String.format(
                        Locale.US,
                        "%s,%s,%d,%d,%s,%d,%d,%.4f,%.6f,%d,%s",
                        fileName,
                        dist.name(),
                        exportN,
                        seed,
                        dist.paramsDescription(),
                        distinctKeys,
                        inversions,
                        invPerN,
                        invRatio,
                        fileSize,
                        sha256
                ));
                manifest.newLine();

                System.out.printf(
                        Locale.US,
                        "  [OK] %-32s | distinct=%-7d | inv=%-12d | inv/N=%-10.2f | ratio=%.4f | %.1f KB%n",
                        fileName, distinctKeys, inversions, invPerN, invRatio, fileSize / 1024.0
                );
            }
        }

        System.out.println("Da tao manifest: " + manifestPath.toAbsolutePath());
    }
}