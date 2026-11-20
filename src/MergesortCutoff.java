import edu.princeton.cs.algs4.Insertion;

public class MergesortCutoff {

    private MergesortCutoff() {}

    private static void merge(Comparable[] a, Comparable[] aux, int lo, int mid, int hi) {
        for (int k = lo; k <= hi; k++) {
            aux[k] = a[k];
        }
        int i = lo, j = mid + 1;
        for (int k = lo; k <= hi; k++) {
            if (i > mid) a[k] = aux[j++];
            else if (j > hi) a[k] = aux[i++];
            else if (less(aux[j], aux[i])) a[k] = aux[j++];
            else a[k] = aux[i++];
        }
    }

    // Cải tiến thuật toán Merge Sort bằng cách sử dụng Insertion Sort cho các mảng nhỏ có độ dài <= CUTOFF.
    private static void sort(Comparable[] a, Comparable[] aux, int lo, int hi, int CUTOFF) {
        if (hi <= lo) return; // Tránh StackOverflow khi CUTOFF = 0

        if (hi <= lo + CUTOFF - 1) {
            Insertion.sort(a, lo, hi+1); // algs4.jar quy ước nửa khoảng [lo, hi) nên phải + 1
            return;
        }
        int mid = lo + (hi - lo) / 2;
        sort(a, aux, lo, mid, CUTOFF);
        sort(a, aux, mid + 1, hi, CUTOFF);
        merge(a, aux, lo, mid, hi);
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    public static void sort(Comparable[] a, int CUTOFF) {
        Comparable[] aux = new Comparable[a.length];
        sort(a, aux, 0, a.length - 1, CUTOFF);
    }
}
