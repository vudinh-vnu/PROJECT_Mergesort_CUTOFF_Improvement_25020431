import edu.princeton.cs.algs4.Stopwatch;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Chương trình chạy thí nghiệm thực nghiệm cho dự án Mergesort CUTOFF.
 * Gọi trực tiếp MergesortCutoff.sort(Comparable[], int) và DataFactory.
 *
 * Cách chạy:
 *   java -Xms1536m -Xmx1536m -XX:+AlwaysPreTouch -cp "bin;lib/algs4.jar" ExperimentRunner
 *   java -Xms1536m -Xmx1536m -XX:+AlwaysPreTouch -cp "bin;lib/algs4.jar" ExperimentRunner --quick
 *   java -Xms1536m -Xmx1536m -XX:+AlwaysPreTouch -cp "bin;lib/algs4.jar" ExperimentRunner --phase=3 --opt-cutoff=16
 */
public final class ExperimentRunner {

    // ==================== CẤU HÌNH THÍ NGHIỆM CHUẨN ====================

    // Lưới 18 mốc CUTOFF: đối chứng (1), giảm nhanh (2,4,6), quét tinh (7..16), đáy phẳng (20..48), phân kỳ (64..128)
    private static final int[] CUTOFF_GRID = {
        1, 2, 4, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128
    };

    // Pha 1A: 4 phân phối tại N chuẩn (1.25M tách trực tiếp mốc 7 và mốc 10 về cỡ mảng con lá)
    private static final int PRIMARY_N = 1_250_000;
    private static final int PRIMARY_TRIALS = 15;

    // Pha 1B: 3 mốc N bổ sung trên RANDOM để triệt tiêu hiệu ứng bậc thang của cây chia đôi
    private static final int[] MULTI_N_SIZES = {1_048_576, 1_500_000, 1_750_000};
    private static final int MULTI_N_TRIALS = 11;

    // Pha 3: Doubling Test (2^16 .. 2^23, an toàn tuyệt đối với 8 GB RAM / Heap 1.5 GB)
    private static final int[] DOUBLING_SIZES = {
        65_536, 131_072, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304, 8_388_608
    };
    private static final DataFactory.Distribution[] DOUBLING_DISTS = {
        DataFactory.Distribution.RANDOM,
        DataFactory.Distribution.PARTIALLY_SORTED
    };

    // Cấu hình JIT Warm-up & chống nhiễu nhiệt
    private static final int WARMUP_N = 200_000;
    private static final int WARMUP_ROUNDS = 5;
    private static final long ORDER_SHUFFLE_SEED = 2026112099L;
    private static final long COOLDOWN_BLOCK_MS = 2000L;
    private static final long GC_SETTLE_MS = 60L;

    private static final Path RESULTS_DIR = Paths.get("results");
    private static final DateTimeFormatter TS_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private ExperimentRunner() {}

    // ==================== WRAPPER ĐẾM PHÉP SO SÁNH (PHA 2) ====================

    /**
     * Wrapper tách biệt hoàn toàn để đếm số lần gọi compareTo()
     * mà KHÔNG chỉnh sửa hay gắn biến đếm vào MergesortCutoff.java.
     */
    private static final class CountingInteger implements Comparable<CountingInteger> {
        private static long compareCount = 0L;
        private final int value;

        CountingInteger(int value) {
            this.value = value;
        }

        static void resetCount() {
            compareCount = 0L;
        }

        static long getCount() {
            return compareCount;
        }

        @Override
        public int compareTo(CountingInteger other) {
            compareCount++;
            return Integer.compare(this.value, other.value);
        }
    }

    // ==================== CẤU HÌNH DÒNG LỆNH ====================

    private static final class RunConfig {
        String phase = "all";       // all | 1 | 2 | 3
        Integer forcedOptCutoff = null;
        boolean quickMode = false;

        int primaryN() { return quickMode ? 50_000 : PRIMARY_N; }
        int primaryTrials() { return quickMode ? 3 : PRIMARY_TRIALS; }
        int[] multiNSizes() { return quickMode ? new int[]{65_536} : MULTI_N_SIZES; }
        int multiNTrials() { return quickMode ? 2 : MULTI_N_TRIALS; }
        int[] doublingSizes() {
            return quickMode
                    ? new int[]{16_384, 32_768, 65_536, 131_072}
                    : DOUBLING_SIZES;
        }
        int warmupN() { return quickMode ? 20_000 : WARMUP_N; }
        int warmupRounds() { return quickMode ? 2 : WARMUP_ROUNDS; }
        long blockCooldownMs() { return quickMode ? 100L : COOLDOWN_BLOCK_MS; }

        int doublingTrialsFor(int n) {
            if (quickMode) return 2;
            if (n >= 8_388_608) return 5;
            if (n >= 4_194_304) return 7;
            return 11;
        }
    }

    // ==================== ĐIỂM VÀO CHÍNH ====================

    public static void main(String[] args) throws Exception {
        RunConfig cfg = parseArgs(args);
        Files.createDirectories(RESULTS_DIR);

        Stopwatch totalStopwatch = new Stopwatch();
        LocalDateTime startTime = LocalDateTime.now();

        System.out.println("=================================================================");
        System.out.println("  THUC NGHIEM MERGESORT CUTOFF — EXPERIMENT RUNNER");
        System.out.println("=================================================================");
        System.out.printf(Locale.US, "Thoi diem bat dau : %s%n", startTime.format(TS_FMT));
        System.out.printf(Locale.US, "Che do chay       : phase=%s | quick=%s | forcedOptCutoff=%s%n",
                cfg.phase, cfg.quickMode, cfg.forcedOptCutoff == null ? "auto" : cfg.forcedOptCutoff);
        printHeapSummary();

        // Ghi trước environment_info.txt với trạng thái RUNNING để đảm bảo không mất metadata
        writeEnvironmentInfo(cfg, startTime, null, -1.0, null, "RUNNING");

        boolean needTiming = cfg.phase.equals("all") || cfg.phase.equals("1") || cfg.phase.equals("3");
        if (needTiming) {
            runJvmWarmup(cfg);
        }

        Integer detectedOptCutoff = cfg.forcedOptCutoff;

        // 1) PHA 1: Quét CUTOFF thời gian thực (thuần Integer -> giữ monomorphic JIT)
        if (cfg.phase.equals("all") || cfg.phase.equals("1")) {
            int autoBest = runPhase1Timing(cfg);
            if (detectedOptCutoff == null) {
                detectedOptCutoff = autoBest;
            }
        }

        // 2) PHA 3: Doubling Test (chạy NGAY SAU Pha 1 trước khi nạp CountingInteger để tránh bimorphic dispatch)
        if (cfg.phase.equals("all") || cfg.phase.equals("3")) {
            int optForDoubling = (detectedOptCutoff != null) ? detectedOptCutoff : 16;
            runPhase3Doubling(cfg, optForDoubling);
        }

        // 3) PHA 2: Đếm số phép so sánh (CountingInteger — chạy cuối cùng, tất định 100%)
        if (cfg.phase.equals("all") || cfg.phase.equals("2")) {
            runPhase2OpsCount(cfg);
        }

        double totalElapsedSec = totalStopwatch.elapsedTime();
        LocalDateTime endTime = LocalDateTime.now();
        writeEnvironmentInfo(cfg, startTime, endTime, totalElapsedSec, detectedOptCutoff, "COMPLETED");

        System.out.println("=================================================================");
        System.out.printf(Locale.US, "HOAN THANH TOAN BO THI NGHIEM! Tong thoi gian: %.2f giay (%.2f phut)%n",
                totalElapsedSec, totalElapsedSec / 60.0);
        System.out.println("Thu muc ket qua: " + RESULTS_DIR.toAbsolutePath());
        System.out.println("=================================================================");
    }

    // ==================== JIT WARM-UP ====================

    private static void runJvmWarmup(RunConfig cfg) {
        System.out.printf(Locale.US, "%n[WARM-UP] Lam nong HotSpot JIT (N=%,d, %d vong x %d CUTOFF)...%n",
                cfg.warmupN(), cfg.warmupRounds(), CUTOFF_GRID.length);
        Stopwatch sw = new Stopwatch();
        Integer[] base = DataFactory.generate(DataFactory.Distribution.RANDOM, cfg.warmupN());
        Integer[] work = new Integer[cfg.warmupN()];

        for (int r = 1; r <= cfg.warmupRounds(); r++) {
            for (int cutoff : CUTOFF_GRID) {
                DataFactory.copyInto(base, work);
                MergesortCutoff.sort(work, cutoff);
                assertSortedOrThrow(work, "WARMUP", cfg.warmupN(), cutoff, r);
            }
        }
        // Giải phóng tham chiếu tránh loitering trước khi vào đo thật
        base = null;
        work = null;
        cleanGcAndPause(cfg.blockCooldownMs());
        System.out.printf(Locale.US, "[WARM-UP] Hoan tat sau %.2f giay.%n", sw.elapsedTime());
    }

    // ==================== PHA 1: QUÉT CUTOFF (timing_raw.csv) ====================

    private static int runPhase1Timing(RunConfig cfg) throws IOException {
        Path outFile = RESULTS_DIR.resolve("timing_raw.csv");
        System.out.println("\n[PHA 1] Quet CUTOFF theo thoi gian thuc -> " + outFile);

        int pN = cfg.primaryN();
        int pTrials = cfg.primaryTrials();
        int[] multiNs = cfg.multiNSizes();
        int mTrials = cfg.multiNTrials();

        int totalSorts = DataFactory.Distribution.values().length * pTrials * CUTOFF_GRID.length
                + multiNs.length * mTrials * CUTOFF_GRID.length;
        int completedSorts = 0;
        Stopwatch phaseSw = new Stopwatch();

        // Lưu kết quả RANDOM tại PRIMARY_N để tự chọn mốc CUTOFF tối ưu (theo Median) cho Pha 3
        Map<Integer, List<Double>> primaryRandomSamples = new HashMap<>();
        for (int c : CUTOFF_GRID) {
            primaryRandomSamples.put(c, new ArrayList<>(pTrials));
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8)) {
            writer.write("distribution,n,cutoff,trial,time_ms");
            writer.newLine();

            // --- Pha 1A: 4 phân phối tại N chuẩn ---
            for (DataFactory.Distribution dist : DataFactory.Distribution.values()) {
                System.out.printf(Locale.US, "  -> [1A] Phan phoi %-16s | N=%,d | %d trials x %d CUTOFF%n",
                        dist.name(), pN, pTrials, CUTOFF_GRID.length);

                Integer[] base = DataFactory.generate(dist, pN);
                Integer[] work = new Integer[pN];

                for (int trial = 1; trial <= pTrials; trial++) {
                    int[] shuffledCutoffs = shuffledCopy(CUTOFF_GRID,
                            ORDER_SHUFFLE_SEED + dist.ordinal() * 10_000L + pN + trial);

                    for (int cutoff : shuffledCutoffs) {
                        DataFactory.copyInto(base, work); // Ngoài khoảng bấm giờ

                        long t0 = System.nanoTime();
                        MergesortCutoff.sort(work, cutoff);
                        long t1 = System.nanoTime();

                        assertSortedOrThrow(work, dist.name(), pN, cutoff, trial); // Ngoài bấm giờ + chống DCE

                        double timeMs = (t1 - t0) / 1_000_000.0;
                        writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.4f",
                                dist.name(), pN, cutoff, trial, timeMs));
                        writer.newLine();

                        if (dist == DataFactory.Distribution.RANDOM) {
                            primaryRandomSamples.get(cutoff).add(timeMs);
                        }
                        completedSorts++;
                    }
                    // Làm sạch nhẹ Young Gen sau mỗi vòng đo (ngoài đồng hồ)
                    cleanGcAndPause(GC_SETTLE_MS);
                    printProgress("Pha 1", completedSorts, totalSorts, phaseSw.elapsedTime());
                }

                writer.flush();
                base = null;
                work = null;
                cleanGcAndPause(cfg.blockCooldownMs());
            }

            // --- Pha 1B: Multi-N trên RANDOM để triệt tiêu hiệu ứng bậc thang ---
            for (int n : multiNs) {
                System.out.printf(Locale.US, "  -> [1B] Multi-N RANDOM           | N=%,d | %d trials x %d CUTOFF%n",
                        n, mTrials, CUTOFF_GRID.length);

                Integer[] base = DataFactory.generate(DataFactory.Distribution.RANDOM, n);
                Integer[] work = new Integer[n];

                for (int trial = 1; trial <= mTrials; trial++) {
                    int[] shuffledCutoffs = shuffledCopy(CUTOFF_GRID,
                            ORDER_SHUFFLE_SEED + 777_000L + n + trial);

                    for (int cutoff : shuffledCutoffs) {
                        DataFactory.copyInto(base, work);

                        long t0 = System.nanoTime();
                        MergesortCutoff.sort(work, cutoff);
                        long t1 = System.nanoTime();

                        assertSortedOrThrow(work, "RANDOM", n, cutoff, trial);

                        double timeMs = (t1 - t0) / 1_000_000.0;
                        writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.4f",
                                DataFactory.Distribution.RANDOM.name(), n, cutoff, trial, timeMs));
                        writer.newLine();
                        completedSorts++;
                    }
                    cleanGcAndPause(GC_SETTLE_MS);
                    printProgress("Pha 1", completedSorts, totalSorts, phaseSw.elapsedTime());
                }

                writer.flush();
                base = null;
                work = null;
                cleanGcAndPause(cfg.blockCooldownMs());
            }
        }

        // Tìm CUTOFF có Median thấp nhất trên RANDOM tại PRIMARY_N
        int bestCutoff = 10;
        double bestMedian = Double.POSITIVE_INFINITY;
        for (int c : CUTOFF_GRID) {
            double med = median(primaryRandomSamples.get(c));
            if (med < bestMedian) {
                bestMedian = med;
                bestCutoff = c;
            }
        }

        double baseMedian = median(primaryRandomSamples.get(1));
        double gainPct = (baseMedian - bestMedian) / baseMedian * 100.0;
        System.out.printf(Locale.US,
                "[PHA 1] Xong sau %.1fs | So bo RANDOM (N=%,d): CUTOFF_opt = %d (Median=%.2f ms vs CUTOFF=1: %.2f ms, giam %.2f%%)%n",
                phaseSw.elapsedTime(), pN, bestCutoff, bestMedian, baseMedian, gainPct);

        return bestCutoff;
    }

    // ==================== PHA 3: DOUBLING TEST (doubling_test.csv) ====================

    private static void runPhase3Doubling(RunConfig cfg, int optCutoff) throws IOException {
        Path outFile = RESULTS_DIR.resolve("doubling_test.csv");

        // Luôn gồm 1 (baseline), 7 (giáo trình cũ), 10 (giáo trình mới), và optCutoff (tự động khử trùng)
        Set<Integer> cutoffSet = new LinkedHashSet<>();
        cutoffSet.add(1);
        cutoffSet.add(7);
        cutoffSet.add(10);
        cutoffSet.add(optCutoff);
        // Nếu optCutoff trùng 7 hoặc 10, bổ sung thêm mốc đối chứng mở rộng 24 để so sánh rõ hệ số a
        if (cutoffSet.size() < 4) {
            cutoffSet.add(24);
        }
        int[] doublingCutoffs = cutoffSet.stream().mapToInt(Integer::intValue).toArray();

        int[] sizes = cfg.doublingSizes();
        System.out.println("\n[PHA 3] Doubling Test (Quy tac ham mu T(N) = a * N^b) -> " + outFile);
        System.out.println("  Cac moc CUTOFF kiem chung: " + Arrays.toString(doublingCutoffs));

        int totalSorts = 0;
        for (int n : sizes) {
            totalSorts += DOUBLING_DISTS.length * doublingCutoffs.length * cfg.doublingTrialsFor(n);
        }
        int completedSorts = 0;
        Stopwatch phaseSw = new Stopwatch();

        try (BufferedWriter writer = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8)) {
            writer.write("distribution,cutoff,n,trial,time_ms");
            writer.newLine();

            for (DataFactory.Distribution dist : DOUBLING_DISTS) {
                for (int n : sizes) {
                    int trials = cfg.doublingTrialsFor(n);
                    System.out.printf(Locale.US, "  -> [Doubling] %-16s | N=%,9d | %d trials x %d CUTOFF%n",
                            dist.name(), n, trials, doublingCutoffs.length);

                    Integer[] base = DataFactory.generate(dist, n);
                    Integer[] work = new Integer[n];

                    for (int trial = 1; trial <= trials; trial++) {
                        int[] shuffled = shuffledCopy(doublingCutoffs,
                                ORDER_SHUFFLE_SEED + 333_000L + dist.ordinal() * 100_000L + n + trial);

                        for (int cutoff : shuffled) {
                            DataFactory.copyInto(base, work);

                            long t0 = System.nanoTime();
                            MergesortCutoff.sort(work, cutoff);
                            long t1 = System.nanoTime();

                            assertSortedOrThrow(work, dist.name(), n, cutoff, trial);

                            double timeMs = (t1 - t0) / 1_000_000.0;
                            writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.4f",
                                    dist.name(), cutoff, n, trial, timeMs));
                            writer.newLine();
                            completedSorts++;
                        }
                        cleanGcAndPause(GC_SETTLE_MS);
                    }

                    writer.flush();
                    // Giải phóng ngay trước khi nhân đôi N để tránh loitering với mảng tới 8.39M phần tử
                    base = null;
                    work = null;
                    cleanGcAndPause(cfg.blockCooldownMs() / 2);
                    printProgress("Pha 3", completedSorts, totalSorts, phaseSw.elapsedTime());
                }
                cleanGcAndPause(cfg.blockCooldownMs());
            }
        }

        System.out.printf(Locale.US, "[PHA 3] Hoan tat sau %.1f giay.%n", phaseSw.elapsedTime());
    }

    // ==================== PHA 2: ĐẾM PHÉP SO SÁNH (ops_count.csv) ====================

    private static void runPhase2OpsCount(RunConfig cfg) throws IOException {
        Path outFile = RESULTS_DIR.resolve("ops_count.csv");
        int n = cfg.primaryN();
        double nLog2N = (n > 1) ? n * (Math.log(n) / Math.log(2.0)) : 0.0;

        System.out.printf(Locale.US, "%n[PHA 2] Dem so phep so sanh (Cost Model, N=%,d) -> %s%n", n, outFile);

        int totalRuns = DataFactory.Distribution.values().length * CUTOFF_GRID.length;
        int completed = 0;
        Stopwatch phaseSw = new Stopwatch();

        try (BufferedWriter writer = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8)) {
            writer.write("distribution,n,cutoff,compares,theoretical_nlogn");
            writer.newLine();

            for (DataFactory.Distribution dist : DataFactory.Distribution.values()) {
                // Sinh từ cùng DataFactory & cùng seed như Pha 1 để ghép cặp 1-1
                Integer[] raw = DataFactory.generate(dist, n);
                CountingInteger[] base = new CountingInteger[n];
                for (int i = 0; i < n; i++) {
                    base[i] = new CountingInteger(raw[i]);
                }
                raw = null; // Giải phóng mảng nguyên thủy ngay
                CountingInteger[] work = new CountingInteger[n];

                for (int cutoff : CUTOFF_GRID) {
                    System.arraycopy(base, 0, work, 0, n);
                    CountingInteger.resetCount();

                    MergesortCutoff.sort(work, cutoff);
                    long compares = CountingInteger.getCount(); // Chốt biến đếm TRƯỚC khi kiểm tra isSorted

                    assertSortedOrThrow(work, "OPS-" + dist.name(), n, cutoff, 1);

                    writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.2f",
                            dist.name(), n, cutoff, compares, nLog2N));
                    writer.newLine();
                    completed++;
                }

                writer.flush();
                base = null;
                work = null;
                cleanGcAndPause(GC_SETTLE_MS);
                printProgress("Pha 2", completed, totalRuns, phaseSw.elapsedTime());
            }
        }

        System.out.printf(Locale.US, "[PHA 2] Hoan tat sau %.1f giay.%n", phaseSw.elapsedTime());
    }

    // ==================== GHI THÔNG TIN MÔI TRƯỜNG ====================

    private static void writeEnvironmentInfo(
            RunConfig cfg,
            LocalDateTime startTime,
            LocalDateTime endTime,
            double totalElapsedSec,
            Integer detectedOptCutoff,
            String status
    ) throws IOException {
        Path envFile = RESULTS_DIR.resolve("environment_info.txt");
        Runtime rt = Runtime.getRuntime();
        RuntimeMXBean mx = ManagementFactory.getRuntimeMXBean();

        try (BufferedWriter w = Files.newBufferedWriter(envFile, StandardCharsets.UTF_8)) {
            w.write("=================================================================\n");
            w.write("EXPERIMENT ENVIRONMENT & REPRODUCIBILITY METADATA\n");
            w.write("=================================================================\n");
            w.write("Status                 : " + status + "\n");
            w.write("Start Time             : " + startTime.format(TS_FMT) + "\n");
            w.write("End Time               : " + (endTime == null ? "(in progress)" : endTime.format(TS_FMT)) + "\n");
            if (totalElapsedSec >= 0) {
                w.write(String.format(Locale.US, "Total Elapsed (algs4)  : %.2f seconds (%.2f minutes)\n",
                        totalElapsedSec, totalElapsedSec / 60.0));
            }
            w.write("\n--- SYSTEM & HARDWARE ---\n");
            w.write("OS Name                : " + System.getProperty("os.name") + "\n");
            w.write("OS Version             : " + System.getProperty("os.version") + "\n");
            w.write("OS Architecture        : " + System.getProperty("os.arch") + "\n");
            w.write("Available Logical Cores: " + rt.availableProcessors() + "\n");

            w.write("\n--- JAVA & JVM ---\n");
            w.write("Java Version           : " + System.getProperty("java.version") + "\n");
            w.write("Java Vendor            : " + System.getProperty("java.vendor") + "\n");
            w.write("JVM Name               : " + System.getProperty("java.vm.name") + "\n");
            w.write("JVM Version            : " + System.getProperty("java.vm.version") + "\n");
            w.write("JVM Input Arguments    : " + mx.getInputArguments() + "\n");
            w.write(String.format(Locale.US, "Heap Max (-Xmx)        : %.2f MB\n", rt.maxMemory() / (1024.0 * 1024.0)));
            w.write(String.format(Locale.US, "Heap Total (Committed) : %.2f MB\n", rt.totalMemory() / (1024.0 * 1024.0)));
            w.write("Assertions Enabled     : " + isAssertionsEnabled() + "\n");

            w.write("\n--- EXPERIMENT PARAMETERS ---\n");
            w.write("Phase Selected         : " + cfg.phase + "\n");
            w.write("Quick Mode             : " + cfg.quickMode + "\n");
            w.write("Timer API (Trials)     : System.nanoTime() -> milliseconds\n");
            w.write("Timer API (Total Run)  : edu.princeton.cs.algs4.Stopwatch\n");
            w.write("Base Data Seed         : " + DataFactory.BASE_SEED + "\n");
            w.write("Order Shuffle Seed     : " + ORDER_SHUFFLE_SEED + "\n");
            w.write("CUTOFF Grid            : " + Arrays.toString(CUTOFF_GRID) + "\n");
            w.write("Phase 1A Primary N     : " + cfg.primaryN() + " (" + cfg.primaryTrials() + " trials/dist)\n");
            w.write("Phase 1B Multi-N       : " + Arrays.toString(cfg.multiNSizes()) + " (" + cfg.multiNTrials() + " trials)\n");
            w.write("Phase 2 Ops Count N    : " + cfg.primaryN() + " (CountingInteger wrapper)\n");
            w.write("Phase 3 Doubling Sizes : " + Arrays.toString(cfg.doublingSizes()) + "\n");
            w.write("Detected Opt CUTOFF    : " + (detectedOptCutoff == null ? "(pending)" : detectedOptCutoff) + "\n");
            w.write("Execution Order        : Warm-up -> Phase 1 -> Phase 3 -> Phase 2 (monomorphic JIT preservation)\n");
        }
    }

    // ==================== HÀM TIỆN ÍCH ====================

    private static RunConfig parseArgs(String[] args) {
        RunConfig cfg = new RunConfig();
        for (String arg : args) {
            if (arg.equals("--quick")) {
                cfg.quickMode = true;
            } else if (arg.startsWith("--phase=")) {
                cfg.phase = arg.substring("--phase=".length()).trim().toLowerCase(Locale.US);
                if (!cfg.phase.equals("all") && !cfg.phase.equals("1")
                        && !cfg.phase.equals("2") && !cfg.phase.equals("3")) {
                    throw new IllegalArgumentException("Tham so --phase khong hop le: " + cfg.phase);
                }
            } else if (arg.startsWith("--opt-cutoff=")) {
                cfg.forcedOptCutoff = Integer.parseInt(arg.substring("--opt-cutoff=".length()).trim());
            } else {
                throw new IllegalArgumentException("Khong nhan dien duoc tham so: " + arg);
            }
        }
        return cfg;
    }

    private static int[] shuffledCopy(int[] source, long seed) {
        int[] copy = Arrays.copyOf(source, source.length);
        Random rng = new Random(seed);
        for (int i = copy.length - 1; i > 0; i--) {
            int r = rng.nextInt(i + 1);
            int tmp = copy[i];
            copy[i] = copy[r];
            copy[r] = tmp;
        }
        return copy;
    }

    private static <T extends Comparable<T>> void assertSortedOrThrow(
            T[] a, String dist, int n, int cutoff, int trial
    ) {
        for (int i = 1; i < a.length; i++) {
            if (a[i].compareTo(a[i - 1]) < 0) {
                throw new IllegalStateException(String.format(
                        Locale.US,
                        "LOI SAP XEP! dist=%s, N=%d, CUTOFF=%d, trial=%d tai vi tri %d",
                        dist, n, cutoff, trial, i
                ));
            }
        }
    }

    private static double median(List<Double> values) {
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int m = sorted.size();
        if (m == 0) return Double.NaN;
        return (m % 2 == 1) ? sorted.get(m / 2) : (sorted.get(m / 2 - 1) + sorted.get(m / 2)) / 2.0;
    }

    private static void cleanGcAndPause(long millis) {
        System.gc();
        if (millis <= 0) return;
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void printProgress(String label, int done, int total, double elapsedSec) {
        double pct = (total > 0) ? (100.0 * done / total) : 100.0;
        double etaSec = (done > 0) ? elapsedSec * (total - done) / done : 0.0;
        int barWidth = 24;
        int filled = (int) Math.round(barWidth * pct / 100.0);
        String bar = "#".repeat(Math.max(0, filled)) + "-".repeat(Math.max(0, barWidth - filled));
        System.out.printf(Locale.US, "     [%s] [%s] %5.1f%% (%d/%d) | Elapsed: %5.1fs | ETA: %5.1fs%n",
                label, bar, pct, done, total, elapsedSec, etaSec);
    }

    private static void printHeapSummary() {
        Runtime rt = Runtime.getRuntime();
        System.out.printf(Locale.US, "JVM Heap          : total=%.0f MB | max=%.0f MB | assertions=%s%n",
                rt.totalMemory() / (1024.0 * 1024.0),
                rt.maxMemory() / (1024.0 * 1024.0),
                isAssertionsEnabled());
        if (isAssertionsEnabled()) {
            System.err.println("CANH BAO: Dang bat -ea (assertions)! Nen tat -ea khi do thoi gian thuc.");
        }
    }

    private static boolean isAssertionsEnabled() {
        boolean enabled = false;
        assert (enabled = true);
        return enabled;
    }
}