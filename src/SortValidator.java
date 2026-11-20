import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SortValidator {

    // Toàn bộ lưới CUTOFF thí nghiệm + mốc phòng thủ 0
    private static final int[] CUTOFF_GRID = {
        0, 1, 2, 4, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128
    };

    private static final long SEED = 20261120L;

    private static int totalChecks = 0;
    private static int failedChecks = 0;

    /**
     * Đối tượng thử nghiệm tính ổn định:
     * So sánh CHỈ theo key, lưu originalIndex để xác minh thứ tự tương đối.
     */
    private static final class StableItem implements Comparable<StableItem> {
        final int key;
        final int originalIndex;

        StableItem(int key, int originalIndex) {
            this.key = key;
            this.originalIndex = originalIndex;
        }

        @Override
        public int compareTo(StableItem other) {
            return Integer.compare(this.key, other.key);
        }

        @Override
        public String toString() {
            return "(" + key + "@" + originalIndex + ")";
        }
    }

    public static void main(String[] args) {
        System.out.println("=== BAT DAU KIEM DINH MergesortCutoff ===");
        System.out.println("Luoi CUTOFF kiem thu (" + CUTOFF_GRID.length + " moc): "
                + Arrays.toString(CUTOFF_GRID));

        runEdgeCaseTests();
        runNonPowerOfTwoSizeTests();
        runDistributionTests();
        runStabilityTests();

        System.out.println("=========================================");
        System.out.println("Tong so ca kiem thu (test cases): " + totalChecks);
        System.out.println("So ca that bai (failed):          " + failedChecks);

        if (failedChecks > 0) {
            System.err.println("KET QUA: THAT BAI! Can sua loi truoc khi chay benchmark.");
            System.exit(1);
        } else {
            System.out.println("KET QUA: DAT 100% (Sorted + Multiset + Stability).");
        }
    }

    // 1. Các trường hợp biên cơ bản & giá trị cực trị của Integer
    private static void runEdgeCaseTests() {
        System.out.println("[1/4] Kiem thu truong hop bien (empty, 1, 2, extremes, all-equal)...");
        Integer[][] edgeCases = {
            {},
            {42},
            {1, 2},
            {2, 1},
            {7, 7},
            {Integer.MAX_VALUE, Integer.MIN_VALUE, 0, -1, 1, Integer.MIN_VALUE, Integer.MAX_VALUE},
            generateAllEqual(50, 99),
            generateAllEqual(501, -5)
        };
        String[] labels = {
            "empty", "single", "two-sorted", "two-reversed", "two-equal",
            "int-extremes", "all-equal-50", "all-equal-501"
        };

        for (int i = 0; i < edgeCases.length; i++) {
            verifyAcrossCutoffs(labels[i], edgeCases[i]);
        }
    }

    // 2. Kích thước lẻ/chẵn không phải lũy thừa của 2, quét quanh các biên CUTOFF
    private static void runNonPowerOfTwoSizeTests() {
        System.out.println("[2/4] Kiem thu kich thuoc le/chan khong phai luy thua cua 2...");
        int[] sizes = {3, 5, 7, 9, 10, 11, 15, 17, 31, 33, 63, 65, 100, 127, 129, 255, 500, 1001};
        Random rng = new Random(SEED + 101);

        for (int n : sizes) {
            Integer[] arr = generateRandomPermutation(n, rng);
            verifyAcrossCutoffs("non-pow2-random-N" + n, arr);
        }
    }

    // 3. Các phân phối chuẩn mực: Ascending, Reverse, Partially-Sorted, Few-Unique, Random
    private static void runDistributionTests() {
        System.out.println("[3/4] Kiem thu cac phan phoi du lieu dau vao...");
        int[] sizes = {100, 503, 2000};
        Random rng = new Random(SEED + 202);

        for (int n : sizes) {
            verifyAcrossCutoffs("ascending-N" + n, generateAscending(n));
            verifyAcrossCutoffs("reverse-N" + n, generateReverse(n));
            verifyAcrossCutoffs("partially-sorted-N" + n, generatePartiallySorted(n, rng));
            verifyAcrossCutoffs("few-unique-5-N" + n, generateFewUnique(n, 5, rng));
            verifyAcrossCutoffs("random-N" + n, generateRandomPermutation(n, rng));
        }
    }

    // 4. Kiểm thử tính ổn định (Stability) với nhiều khóa trùng lặp
    private static void runStabilityTests() {
        System.out.println("[4/4] Kiem thu tinh on dinh (Stability Test)...");
        int[] sizes = {2, 7, 10, 15, 33, 128, 500, 2000};
        int[] keyCounts = {2, 5, 10, 50};
        Random rng = new Random(SEED + 303);

        for (int n : sizes) {
            for (int k : keyCounts) {
                // Mẫu A: khóa ngẫu nhiên nhiều trùng lặp
                StableItem[] randomItems = new StableItem[n];
                for (int i = 0; i < n; i++) {
                    randomItems[i] = new StableItem(rng.nextInt(k), i);
                }
                verifyStabilityAcrossCutoffs("stable-random-N" + n + "-K" + k, randomItems);

                // Mẫu B: khóa giảm dần có lặp (kích hoạt đảo vị trí cực đại ở Insertion & Merge)
                StableItem[] reverseKeyItems = new StableItem[n];
                for (int i = 0; i < n; i++) {
                    reverseKeyItems[i] = new StableItem((n - 1 - i) % k, i);
                }
                verifyStabilityAcrossCutoffs("stable-revkey-N" + n + "-K" + k, reverseKeyItems);
            }
        }
    }

    // ==================== CORE VERIFIERS ====================

    private static void verifyAcrossCutoffs(String caseName, Integer[] original) {
        for (int cutoff : CUTOFF_GRID) {
            totalChecks++;
            Integer[] work = Arrays.copyOf(original, original.length);
            try {
                MergesortCutoff.sort(work, cutoff);
            } catch (Throwable t) {
                reportFailure(caseName, cutoff, "Exception: " + t);
                continue;
            }

            if (!isSorted(work)) {
                reportFailure(caseName, cutoff, "Mang dau ra KHONG tang dan (isSorted = false)");
                continue;
            }
            if (!sameMultiset(original, work)) {
                reportFailure(caseName, cutoff, "Vi pham bao toan da tap (multiset mismatch)");
            }
        }
    }

    private static void verifyStabilityAcrossCutoffs(String caseName, StableItem[] original) {
        for (int cutoff : CUTOFF_GRID) {
            totalChecks++;
            StableItem[] work = Arrays.copyOf(original, original.length);
            try {
                MergesortCutoff.sort(work, cutoff);
            } catch (Throwable t) {
                reportFailure(caseName, cutoff, "Exception: " + t);
                continue;
            }

            // (a) Kiểm tra thứ tự khóa không giảm
            if (!isSorted(work)) {
                reportFailure(caseName, cutoff, "Stability case: mang KHONG tang dan theo key");
                continue;
            }

            // (b) Kiểm tra bảo toàn đúng 100% đối tượng gốc (mỗi originalIndex xuất hiện đúng 1 lần & cùng tham chiếu)
            boolean[] seen = new boolean[original.length];
            boolean preservationOk = true;
            for (StableItem item : work) {
                if (item == null || item.originalIndex < 0 || item.originalIndex >= original.length
                        || seen[item.originalIndex] || original[item.originalIndex] != item) {
                    preservationOk = false;
                    break;
                }
                seen[item.originalIndex] = true;
            }
            if (!preservationOk) {
                reportFailure(caseName, cutoff, "Stability case: mat/trung doi tuong (multiset mismatch)");
                continue;
            }

            // (c) Kiểm tra tính ổn định: khóa bằng nhau => originalIndex phải tăng ngặt
            for (int i = 1; i < work.length; i++) {
                if (work[i - 1].key == work[i].key && work[i - 1].originalIndex >= work[i].originalIndex) {
                    reportFailure(caseName, cutoff,
                            "Vi pham tinh on dinh tai vi tri " + i + ": "
                            + work[i - 1] + " dung truoc " + work[i]);
                    break;
                }
            }
        }
    }

    private static <T extends Comparable<T>> boolean isSorted(T[] a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i].compareTo(a[i - 1]) < 0) return false;
        }
        return true;
    }

    private static <T> boolean sameMultiset(T[] before, T[] after) {
        if (before.length != after.length) return false;
        Map<T, Integer> counts = new HashMap<>();
        for (T x : before) {
            counts.merge(x, 1, Integer::sum);
        }
        for (T x : after) {
            Integer c = counts.get(x);
            if (c == null || c == 0) return false;
            if (c == 1) counts.remove(x);
            else counts.put(x, c - 1);
        }
        return counts.isEmpty();
    }

    private static void reportFailure(String caseName, int cutoff, String reason) {
        failedChecks++;
        System.err.printf("  [FAIL] case=%s | CUTOFF=%d | %s%n", caseName, cutoff, reason);
    }

    // ==================== DATA GENERATORS ====================

    private static Integer[] generateAllEqual(int n, int val) {
        Integer[] a = new Integer[n];
        Arrays.fill(a, val);
        return a;
    }

    private static Integer[] generateAscending(int n) {
        Integer[] a = new Integer[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    private static Integer[] generateReverse(int n) {
        Integer[] a = new Integer[n];
        for (int i = 0; i < n; i++) a[i] = n - 1 - i;
        return a;
    }

    private static Integer[] generateRandomPermutation(int n, Random rng) {
        Integer[] a = generateAscending(n);
        // Knuth Shuffle
        for (int i = n - 1; i > 0; i--) {
            int r = rng.nextInt(i + 1);
            Integer tmp = a[i];
            a[i] = a[r];
            a[r] = tmp;
        }
        return a;
    }

    private static Integer[] generatePartiallySorted(int n, Random rng) {
        Integer[] a = generateAscending(n);
        int window = 8;
        int swaps = Math.max(1, n / 10);
        if (n <= 1) return a;
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

    private static Integer[] generateFewUnique(int n, int numKeys, Random rng) {
        Integer[] a = new Integer[n];
        for (int i = 0; i < n; i++) a[i] = rng.nextInt(numKeys);
        return a;
    }
}