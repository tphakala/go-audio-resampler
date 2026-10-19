/*
 * Quality measurement tool for soxr reference
 *
 * Measures various audio quality metrics:
 * - Passband ripple (gain flatness of 20 tones across the passband)
 * - THD (Total Harmonic Distortion)
 * - SNR (Signal-to-Noise Ratio)
 * - Impulse response characteristics
 *
 * Compiles with: gcc -o test_quality test_quality.c -lsoxr -lm -lfftw3
 * Usage: ./test_quality <input_rate> <output_rate> <test_type>
 *
 * test_type:
 *   ripple[:band]  Passband gain flatness of 20 tones up to band times the
 *                  lower Nyquist (default band 0.9)
 *   thd:1000    - THD at 1000 Hz
 *   snr:1000    - SNR with 1000 Hz test tone
 *   impulse     - Impulse response analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <soxr.h>
#include <fftw3.h>

#define QUALITY_SAMPLES 65536
#define FFT_SIZE 16384
#define M_PI 3.14159265358979323846

/* Perform FFT and return magnitude spectrum in dB */
/* If use_window is 0, no window is applied (for impulse response analysis) */
void compute_spectrum_ex(const double *signal, size_t signal_len, double sample_rate,
                         double *mag_db, double *freqs, size_t fft_size, int use_window) {

    fftw_complex *fft_in = fftw_alloc_complex(fft_size);
    fftw_complex *fft_out = fftw_alloc_complex(fft_size);
    fftw_plan plan = fftw_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    /* Copy to FFT input with optional Hann window */
    for (size_t i = 0; i < fft_size; i++) {
        double window = use_window ? 0.5 * (1.0 - cos(2.0 * M_PI * i / (fft_size - 1))) : 1.0;
        double val = (i < signal_len) ? signal[i] : 0.0;
        fft_in[i][0] = val * window;
        fft_in[i][1] = 0.0;
    }

    fftw_execute(plan);

    /* Compute magnitude in dB */
    size_t num_bins = fft_size / 2 + 1;
    for (size_t k = 0; k < num_bins; k++) {
        double re = fft_out[k][0];
        double im = fft_out[k][1];
        double mag = sqrt(re * re + im * im);
        mag_db[k] = 20.0 * log10(mag + 1e-20);
        freqs[k] = (double)k * sample_rate / fft_size;
    }

    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

/* Wrapper with window (for sine wave analysis) */
void compute_spectrum(const double *signal, size_t signal_len, double sample_rate,
                      double *mag_db, double *freqs, size_t fft_size) {
    compute_spectrum_ex(signal, signal_len, sample_rate, mag_db, freqs, fft_size, 1);
}

/* Quality preset mapping */
typedef enum {
    QUALITY_QUICK = 0,
    QUALITY_LOW = 1,
    QUALITY_MEDIUM = 2,
    QUALITY_HIGH = 3,
    QUALITY_VERY_HIGH = 4
} quality_preset_t;

/* Global quality setting (default to VHQ for backwards compatibility) */
static unsigned long g_soxr_quality = SOXR_VHQ;

/* Set quality preset */
void set_quality_preset(quality_preset_t preset) {
    switch (preset) {
        case QUALITY_QUICK:     g_soxr_quality = SOXR_QQ; break;
        case QUALITY_LOW:       g_soxr_quality = SOXR_LQ; break;
        case QUALITY_MEDIUM:    g_soxr_quality = SOXR_MQ; break;
        case QUALITY_HIGH:      g_soxr_quality = SOXR_HQ; break;
        case QUALITY_VERY_HIGH: g_soxr_quality = SOXR_VHQ; break;
        default:                g_soxr_quality = SOXR_VHQ; break;
    }
}

/* Get quality name for logging */
const char* get_quality_name(void) {
    switch (g_soxr_quality) {
        case SOXR_QQ:  return "Quick";
        case SOXR_LQ:  return "Low";
        case SOXR_MQ:  return "Medium";
        case SOXR_HQ:  return "High";
        case SOXR_VHQ: return "VeryHigh";
        default:       return "Unknown";
    }
}

/* Resample using soxr with configurable quality */
double *resample_soxr(const double *input, size_t input_len,
                      double input_rate, double output_rate, size_t *output_len) {

    soxr_quality_spec_t q_spec = soxr_quality_spec(g_soxr_quality, 0);
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);

    size_t expected_output = (size_t)(input_len * output_rate / input_rate + 1024);
    double *output = malloc(expected_output * sizeof(double));

    size_t idone, odone;
    soxr_error_t error = soxr_oneshot(input_rate, output_rate, 1,
                                       input, input_len, &idone,
                                       output, expected_output, &odone,
                                       &io_spec, &q_spec, NULL);

    if (error) {
        fprintf(stderr, "soxr error: %s\n", error);
        free(output);
        return NULL;
    }

    *output_len = odone;
    return output;
}

/*
 * Level in dB (re an amplitude of 1) of the component at frequency f in y,
 * sampled at fs. Line-for-line port of ToneLevelDB in
 * internal/testutil/tone.go; keep the two identical. Windows the middle
 * segment (at most 16384 samples) with a 4-term Blackman-Harris window and
 * evaluates a single DFT bin at f. Returns -INFINITY when y is too short.
 */
static double tone_level_db(const double *y, size_t len, double fs, double f) {
    const long max_segment = 16384;
    const long edge_margin = 1000;
    long total = (long)len;
    long n = max_segment;
    if (total < n + 2 * edge_margin) {
        n = total - 2 * edge_margin;
    }
    if (n < 2) {
        return -INFINITY;
    }
    long start = (total - n) / 2;
    double re = 0.0, im = 0.0, ws = 0.0;
    for (long i = 0; i < n; i++) {
        double x = 2.0 * M_PI * (double)i / (double)(n - 1);
        double w = 0.35875 - 0.48829 * cos(x) + 0.14128 * cos(2.0 * x) - 0.01168 * cos(3.0 * x);
        ws += w;
        double ph = 2.0 * M_PI * f * (double)i / fs;
        double v = y[start + i] * w;
        re += v * cos(ph);
        im -= v * sin(ph);
    }
    double amp = 2.0 * hypot(re, im) / ws;
    return 20.0 * log10(amp + 1e-300);
}

#define RIPPLE_NUM_TONES 20
#define RIPPLE_LOWEST_HZ 500.0
#define RIPPLE_TONE_AMP 0.05

/*
 * Measure passband gain flatness. Must match measurePassbandRipple in
 * internal/engine/quality_comparison_test.go: 20 equal-amplitude tones from
 * 500 Hz to band_frac * min(input_rate, output_rate) / 2 (the last one exactly
 * at the band end), one-shot resample, per-tone gain read with tone_level_db
 * from the middle of the output and referenced to the input amplitude.
 * ripple = max gain - min gain. Changing it means regenerating the ripple_*
 * entries of internal/engine/testdata/soxr_reference_data.json.
 */
void measure_ripple(double input_rate, double output_rate, double band_frac) {
    double orig_nyquist = (input_rate < output_rate) ? input_rate / 2.0 : output_rate / 2.0;
    double hi = band_frac * orig_nyquist;

    if (!(hi > RIPPLE_LOWEST_HZ)) {
        fprintf(stderr, "ripple band end %.1f Hz is not above the lowest tone %.0f Hz\n",
                hi, RIPPLE_LOWEST_HZ);
        exit(1);
    }

    double test_freqs[RIPPLE_NUM_TONES];
    for (int k = 0; k < RIPPLE_NUM_TONES; k++) {
        test_freqs[k] = RIPPLE_LOWEST_HZ + k * (hi - RIPPLE_LOWEST_HZ) / (RIPPLE_NUM_TONES - 1);
    }

    double *input = calloc(QUALITY_SAMPLES, sizeof(double));
    for (int i = 0; i < RIPPLE_NUM_TONES; i++) {
        for (size_t s = 0; s < QUALITY_SAMPLES; s++) {
            double phase = 2.0 * M_PI * test_freqs[i] * s / input_rate;
            input[s] += RIPPLE_TONE_AMP * sin(phase);
        }
    }

    /* Resample */
    size_t output_len;
    double *output = resample_soxr(input, QUALITY_SAMPLES, input_rate, output_rate, &output_len);
    free(input);

    if (!output) {
        fprintf(stderr, "Resampling failed\n");
        exit(1);
    }

    double ref_db = 20.0 * log10(RIPPLE_TONE_AMP);
    double max_gain = -INFINITY;
    double min_gain = INFINITY;
    for (int i = 0; i < RIPPLE_NUM_TONES; i++) {
        double gain = tone_level_db(output, output_len, output_rate, test_freqs[i]) - ref_db;
        if (!isfinite(gain)) {
            fprintf(stderr, "tone %.1f Hz has no finite level (output has %zu samples)\n",
                    test_freqs[i], output_len);
            free(output);
            exit(1);
        }
        if (gain > max_gain) max_gain = gain;
        if (gain < min_gain) min_gain = gain;
    }
    free(output);

    printf("# Passband Ripple Test Results\n");
    printf("# ============================\n");
    printf("# quality = %s\n", get_quality_name());
    printf("# input_rate = %.0f\n", input_rate);
    printf("# output_rate = %.0f\n", output_rate);
    printf("# band_fraction = %.6f\n", band_frac);
    printf("# test_freqs = %d tones from %.0f to %.0f Hz\n", RIPPLE_NUM_TONES, RIPPLE_LOWEST_HZ, hi);
    printf("# max_deviation = %.6f dB\n", max_gain);
    printf("# min_deviation = %.6f dB\n", min_gain);
    printf("# ripple = %.6f dB\n", max_gain - min_gain);
}

/* Measure THD */
void measure_thd(double input_rate, double output_rate, double test_freq) {
    /* Generate pure sine */
    double *input = malloc(QUALITY_SAMPLES * sizeof(double));
    for (size_t i = 0; i < QUALITY_SAMPLES; i++) {
        double phase = 2.0 * M_PI * test_freq * i / input_rate;
        input[i] = 0.9 * sin(phase);
    }

    /* Resample */
    size_t output_len;
    double *output = resample_soxr(input, QUALITY_SAMPLES, input_rate, output_rate, &output_len);
    free(input);

    if (!output) {
        fprintf(stderr, "Resampling failed\n");
        return;
    }

    /* Compute spectrum */
    size_t num_bins = FFT_SIZE / 2 + 1;
    double *mag_db = malloc(num_bins * sizeof(double));
    double *freqs = malloc(num_bins * sizeof(double));
    compute_spectrum(output, output_len, output_rate, mag_db, freqs, FFT_SIZE);
    free(output);

    /* Find fundamental */
    size_t fundamental_bin = (size_t)(test_freq / output_rate * FFT_SIZE);
    double fundamental_mag = pow(10.0, mag_db[fundamental_bin] / 20.0);

    /* Find harmonics */
    double harmonic_power_sum = 0.0;
    double nyquist = output_rate / 2.0;

    for (int h = 2; h <= 10; h++) {
        double harm_freq = test_freq * h;
        if (harm_freq >= nyquist) break;

        size_t harm_bin = (size_t)(harm_freq / output_rate * FFT_SIZE);
        if (harm_bin < num_bins) {
            double harm_mag = pow(10.0, mag_db[harm_bin] / 20.0);
            harmonic_power_sum += harm_mag * harm_mag;
        }
    }

    double thd_ratio = sqrt(harmonic_power_sum) / (fundamental_mag + 1e-20);
    double thd_db = 20.0 * log10(thd_ratio + 1e-20);
    double thd_percent = thd_ratio * 100.0;

    printf("# THD Test Results\n");
    printf("# ================\n");
    printf("# quality = %s\n", get_quality_name());
    printf("# input_rate = %.0f\n", input_rate);
    printf("# output_rate = %.0f\n", output_rate);
    printf("# test_freq = %.0f\n", test_freq);
    printf("# thd_db = %.6f dB\n", thd_db);
    printf("# thd_percent = %.8f %%\n", thd_percent);

    free(mag_db);
    free(freqs);
}

/* Measure SNR */
void measure_snr(double input_rate, double output_rate, double test_freq) {
    /* Generate pure sine */
    double *input = malloc(QUALITY_SAMPLES * sizeof(double));
    for (size_t i = 0; i < QUALITY_SAMPLES; i++) {
        double phase = 2.0 * M_PI * test_freq * i / input_rate;
        input[i] = 0.9 * sin(phase);
    }

    /* Resample */
    size_t output_len;
    double *output = resample_soxr(input, QUALITY_SAMPLES, input_rate, output_rate, &output_len);
    free(input);

    if (!output) {
        fprintf(stderr, "Resampling failed\n");
        return;
    }

    /* Compute spectrum */
    size_t num_bins = FFT_SIZE / 2 + 1;
    double *mag_db = malloc(num_bins * sizeof(double));
    double *freqs = malloc(num_bins * sizeof(double));
    compute_spectrum(output, output_len, output_rate, mag_db, freqs, FFT_SIZE);
    free(output);

    /* Find signal power (fundamental + nearby bins) */
    size_t fundamental_bin = (size_t)(test_freq / output_rate * FFT_SIZE);
    double signal_power = 0.0;
    for (int b = -2; b <= 2; b++) {
        size_t bin = fundamental_bin + b;
        if (bin > 0 && bin < num_bins) {
            double mag = pow(10.0, mag_db[bin] / 20.0);
            signal_power += mag * mag;
        }
    }

    /* Find noise power (everything else) */
    double noise_power = 0.0;
    for (size_t b = 1; b < num_bins; b++) {
        /* Skip signal bins and harmonics */
        int is_signal = 0;
        for (int h = 1; h <= 10; h++) {
            size_t harm_bin = fundamental_bin * h;
            if (b >= harm_bin - 2 && b <= harm_bin + 2) {
                is_signal = 1;
                break;
            }
        }
        if (!is_signal) {
            double mag = pow(10.0, mag_db[b] / 20.0);
            noise_power += mag * mag;
        }
    }

    double signal_db = 10.0 * log10(signal_power + 1e-20);
    double noise_db = 10.0 * log10(noise_power + 1e-20);
    double snr_db = signal_db - noise_db;

    printf("# SNR Test Results\n");
    printf("# ================\n");
    printf("# quality = %s\n", get_quality_name());
    printf("# input_rate = %.0f\n", input_rate);
    printf("# output_rate = %.0f\n", output_rate);
    printf("# test_freq = %.0f\n", test_freq);
    printf("# signal_db = %.6f dB\n", signal_db);
    printf("# noise_db = %.6f dB\n", noise_db);
    printf("# snr_db = %.6f dB\n", snr_db);

    free(mag_db);
    free(freqs);
}

/* Measure impulse response */
void measure_impulse(double input_rate, double output_rate) {
    /* Generate impulse in center of buffer */
    size_t num_samples = 8192;
    size_t impulse_pos = num_samples / 2;
    double *impulse = calloc(num_samples, sizeof(double));
    impulse[impulse_pos] = 1.0;

    /* Resample */
    size_t output_len;
    double *output = resample_soxr(impulse, num_samples, input_rate, output_rate, &output_len);
    free(impulse);

    if (!output) {
        fprintf(stderr, "Resampling failed\n");
        return;
    }

    /* Find main peak */
    size_t main_peak_idx = 0;
    double main_peak_val = 0.0;
    for (size_t i = 0; i < output_len; i++) {
        if (fabs(output[i]) > main_peak_val) {
            main_peak_val = fabs(output[i]);
            main_peak_idx = i;
        }
    }

    /* Measure pre-ringing */
    double pre_ringing_peak = 0.0;
    for (size_t i = 0; i < main_peak_idx; i++) {
        if (fabs(output[i]) > pre_ringing_peak) {
            pre_ringing_peak = fabs(output[i]);
        }
    }
    double pre_ringing_db = 20.0 * log10(pre_ringing_peak / main_peak_val + 1e-20);

    /* Measure post-ringing */
    double post_ringing_peak = 0.0;
    for (size_t i = main_peak_idx + 10; i < output_len; i++) {
        if (fabs(output[i]) > post_ringing_peak) {
            post_ringing_peak = fabs(output[i]);
        }
    }
    double post_ringing_db = 20.0 * log10(post_ringing_peak / main_peak_val + 1e-20);

    /* Find ringout time */
    double threshold = main_peak_val * pow(10.0, -60.0 / 20.0);
    size_t ringout_samples = 0;
    for (size_t i = main_peak_idx; i < output_len; i++) {
        if (fabs(output[i]) > threshold) {
            ringout_samples = i - main_peak_idx;
        }
    }

    printf("# Impulse Response Test Results\n");
    printf("# =============================\n");
    printf("# quality = %s\n", get_quality_name());
    printf("# input_rate = %.0f\n", input_rate);
    printf("# output_rate = %.0f\n", output_rate);
    printf("# pre_ringing_db = %.6f dB\n", pre_ringing_db);
    printf("# post_ringing_db = %.6f dB\n", post_ringing_db);
    printf("# ringout_samples = %zu\n", ringout_samples);
    printf("# main_peak_idx = %zu\n", main_peak_idx);

    free(output);
}

/* Measure throughput in samples per second */
void measure_throughput(double input_rate, double output_rate, size_t num_samples, int iterations) {
    /* Generate test signal (sine wave) */
    double *input = malloc(num_samples * sizeof(double));
    for (size_t i = 0; i < num_samples; i++) {
        double phase = 2.0 * M_PI * 1000.0 * i / input_rate;
        input[i] = 0.9 * sin(phase);
    }

    /* Warm up */
    size_t warmup_len;
    double *warmup = resample_soxr(input, num_samples, input_rate, output_rate, &warmup_len);
    if (warmup) free(warmup);

    /* Time multiple iterations */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    size_t total_input_samples = 0;
    size_t total_output_samples = 0;

    for (int i = 0; i < iterations; i++) {
        size_t output_len;
        double *output = resample_soxr(input, num_samples, input_rate, output_rate, &output_len);
        if (output) {
            total_input_samples += num_samples;
            total_output_samples += output_len;
            free(output);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double input_samples_per_sec = total_input_samples / elapsed_sec;
    double output_samples_per_sec = total_output_samples / elapsed_sec;
    double megasamples_per_sec = input_samples_per_sec / 1e6;

    printf("# Throughput Test Results\n");
    printf("# =======================\n");
    printf("# quality = %s\n", get_quality_name());
    printf("# input_rate = %.0f\n", input_rate);
    printf("# output_rate = %.0f\n", output_rate);
    printf("# samples_per_iteration = %zu\n", num_samples);
    printf("# iterations = %d\n", iterations);
    printf("# total_input_samples = %zu\n", total_input_samples);
    printf("# total_output_samples = %zu\n", total_output_samples);
    printf("# elapsed_sec = %.6f\n", elapsed_sec);
    printf("# input_samples_per_sec = %.0f\n", input_samples_per_sec);
    printf("# output_samples_per_sec = %.0f\n", output_samples_per_sec);
    printf("# megasamples_per_sec = %.3f\n", megasamples_per_sec);

    free(input);
}

/* Parse quality string to preset */
quality_preset_t parse_quality(const char *str) {
    if (strcmp(str, "quick") == 0 || strcmp(str, "0") == 0) return QUALITY_QUICK;
    if (strcmp(str, "low") == 0 || strcmp(str, "1") == 0) return QUALITY_LOW;
    if (strcmp(str, "medium") == 0 || strcmp(str, "2") == 0) return QUALITY_MEDIUM;
    if (strcmp(str, "high") == 0 || strcmp(str, "3") == 0) return QUALITY_HIGH;
    if (strcmp(str, "veryhigh") == 0 || strcmp(str, "4") == 0) return QUALITY_VERY_HIGH;
    return QUALITY_VERY_HIGH; /* default */
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_rate> <output_rate> <test_type> [quality]\n", argv[0]);
        fprintf(stderr, "\nTest types:\n");
        fprintf(stderr, "  ripple[:band]  Passband gain flatness of 20 tones up to band times the lower Nyquist (default 0.9)\n");
        fprintf(stderr, "  thd:1000     - THD at 1000 Hz\n");
        fprintf(stderr, "  snr:1000     - SNR with 1000 Hz tone\n");
        fprintf(stderr, "  impulse      - Impulse response analysis\n");
        fprintf(stderr, "  throughput   - Throughput measurement (samples/sec)\n");
        fprintf(stderr, "\nQuality presets (optional, default=veryhigh):\n");
        fprintf(stderr, "  quick/0      - Quick (SOXR_QQ)\n");
        fprintf(stderr, "  low/1        - Low (SOXR_LQ)\n");
        fprintf(stderr, "  medium/2     - Medium (SOXR_MQ)\n");
        fprintf(stderr, "  high/3       - High (SOXR_HQ)\n");
        fprintf(stderr, "  veryhigh/4   - Very High (SOXR_VHQ)\n");
        return 1;
    }

    double input_rate = atof(argv[1]);
    double output_rate = atof(argv[2]);
    char *test_type = argv[3];

    /* Set quality if provided */
    if (argc >= 5) {
        set_quality_preset(parse_quality(argv[4]));
    }

    if (strcmp(test_type, "ripple") == 0) {
        measure_ripple(input_rate, output_rate, 0.9);
    } else if (strncmp(test_type, "ripple:", 7) == 0) {
        char *end = NULL;
        double band = strtod(test_type + 7, &end);
        if (end == test_type + 7 || *end != '\0' || !(band > 0.0 && band <= 1.0)) {
            fprintf(stderr, "Invalid ripple band: %s (want a fraction in (0, 1])\n", test_type + 7);
            return 1;
        }
        measure_ripple(input_rate, output_rate, band);
    } else if (strncmp(test_type, "thd:", 4) == 0) {
        double test_freq = atof(test_type + 4);
        measure_thd(input_rate, output_rate, test_freq);
    } else if (strncmp(test_type, "snr:", 4) == 0) {
        double test_freq = atof(test_type + 4);
        measure_snr(input_rate, output_rate, test_freq);
    } else if (strcmp(test_type, "impulse") == 0) {
        measure_impulse(input_rate, output_rate);
    } else if (strcmp(test_type, "throughput") == 0) {
        /* Use 1 second of audio at input rate, 100 iterations */
        size_t num_samples = (size_t)input_rate;
        measure_throughput(input_rate, output_rate, num_samples, 100);
    } else {
        fprintf(stderr, "Unknown test type: %s\n", test_type);
        return 1;
    }

    return 0;
}
