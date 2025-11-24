/*
 * Anti-aliasing/anti-imaging test for soxr reference
 *
 * This test measures how well the resampler attenuates frequencies:
 *
 * UPSAMPLING (anti-imaging):
 *   For 48kHz -> 96kHz:
 *   - Original Nyquist: 24kHz
 *   - New Nyquist: 48kHz
 *   - The 24-48kHz band should be heavily attenuated (no images)
 *
 * DOWNSAMPLING (anti-aliasing):
 *   For 48kHz -> 32kHz (BirdNET/Perch use case):
 *   - Original Nyquist: 24kHz
 *   - New Nyquist: 16kHz
 *   - Input frequencies 16-24kHz should be attenuated before decimation
 *   - Otherwise they alias back into 0-8kHz
 *
 * Test signal: Broadband noise or swept sine containing all frequencies
 * Measurement: Power spectral density via FFT
 *
 * Compiles with: gcc -o test_antialiasing test_antialiasing.c -lsoxr -lm -lfftw3
 * Usage: ./test_antialiasing <input_rate> <output_rate> <signal_type>
 *
 * signal_type:
 *   noise    - White noise (broadband)
 *   multitone - Multiple sine tones at various frequencies
 *   sweep    - Linear frequency sweep (chirp)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <soxr.h>
#include <fftw3.h>

#define INPUT_SAMPLES 32768  /* Power of 2 for efficient FFT */
#define FFT_SIZE 8192        /* FFT window size */
#define M_PI 3.14159265358979323846

typedef enum {
    SIGNAL_NOISE,
    SIGNAL_MULTITONE,
    SIGNAL_SWEEP,
    SIGNAL_ALIAS_TONES  /* Tones in aliasing region for downsampling tests */
} signal_type_t;

/* Simple linear congruential generator for reproducible "random" noise */
static unsigned int rand_state = 12345;

static double rand_uniform(void) {
    rand_state = rand_state * 1103515245 + 12345;
    return (double)(rand_state & 0x7FFFFFFF) / (double)0x7FFFFFFF * 2.0 - 1.0;
}

/* Generate test signal */
void generate_signal(double *buffer, size_t num_samples, signal_type_t type,
                     double sample_rate) {
    size_t i;

    switch (type) {
        case SIGNAL_NOISE:
            /* White noise - contains all frequencies */
            rand_state = 12345;  /* Reset for reproducibility */
            for (i = 0; i < num_samples; i++) {
                buffer[i] = rand_uniform() * 0.5;
            }
            break;

        case SIGNAL_MULTITONE:
            /* Multiple tones at various frequencies up to Nyquist */
            /* Frequencies: 1k, 2k, 4k, 8k, 12k, 16k, 20k, 22k, 23k Hz */
            {
                double freqs[] = {1000, 2000, 4000, 8000, 12000, 16000, 20000, 22000, 23000};
                int num_freqs = 9;
                double nyquist = sample_rate / 2.0;

                for (i = 0; i < num_samples; i++) {
                    buffer[i] = 0.0;
                }

                for (int f = 0; f < num_freqs; f++) {
                    if (freqs[f] < nyquist * 0.95) {  /* Stay below Nyquist */
                        double amplitude = 0.1;  /* Equal amplitude for each tone */
                        for (i = 0; i < num_samples; i++) {
                            double phase = 2.0 * M_PI * freqs[f] * i / sample_rate;
                            buffer[i] += amplitude * sin(phase);
                        }
                    }
                }
            }
            break;

        case SIGNAL_SWEEP:
            /* Linear frequency sweep from 100 Hz to near-Nyquist */
            {
                double f_start = 100.0;
                double f_end = sample_rate * 0.45;  /* 90% of Nyquist */
                double duration = (double)num_samples / sample_rate;
                double sweep_rate = (f_end - f_start) / duration;

                for (i = 0; i < num_samples; i++) {
                    double t = (double)i / sample_rate;
                    double phase = 2.0 * M_PI * (f_start * t + sweep_rate * t * t / 2.0);
                    buffer[i] = 0.7 * sin(phase);
                }
            }
            break;

        case SIGNAL_ALIAS_TONES:
            /*
             * Tones specifically in the aliasing region for downsampling tests.
             * For 48kHz -> 32kHz: output Nyquist = 16kHz
             * Place tones at 17, 18, 19, 20, 21, 22, 23 kHz
             * These would alias to 1, 2, 3, 4, 5, 6, 7 kHz if not filtered
             */
            {
                double output_nyquist_est = sample_rate / 3.0;  /* Estimate for 48->32 */
                double nyquist = sample_rate / 2.0;

                /* Generate tones from output_nyquist to input_nyquist */
                for (double freq = output_nyquist_est + 1000; freq < nyquist - 500; freq += 1000) {
                    double amplitude = 0.1;
                    for (i = 0; i < num_samples; i++) {
                        double phase = 2.0 * M_PI * freq * i / sample_rate;
                        buffer[i] += amplitude * sin(phase);
                    }
                }
            }
            break;
    }
}

/* Compute power spectral density using FFT */
/* Returns power in dB for each frequency bin */
void compute_psd(const double *signal, size_t signal_len, double sample_rate,
                 double *psd_db, double *freqs, size_t fft_size) {

    fftw_complex *fft_in, *fft_out;
    fftw_plan plan;

    fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    plan = fftw_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    /* Initialize output arrays */
    size_t num_bins = fft_size / 2 + 1;
    for (size_t k = 0; k < num_bins; k++) {
        psd_db[k] = -200.0;  /* Initialize to very low value */
        freqs[k] = (double)k * sample_rate / fft_size;
    }

    /* Use Welch's method - average multiple overlapping windows */
    size_t num_windows = 0;
    double *psd_accum = calloc(num_bins, sizeof(double));

    /* Hann window for spectral leakage reduction */
    double *window = malloc(fft_size * sizeof(double));
    for (size_t n = 0; n < fft_size; n++) {
        window[n] = 0.5 * (1.0 - cos(2.0 * M_PI * n / (fft_size - 1)));
    }

    /* 50% overlap */
    size_t hop_size = fft_size / 2;

    for (size_t start = 0; start + fft_size <= signal_len; start += hop_size) {
        /* Apply window and copy to FFT input */
        for (size_t n = 0; n < fft_size; n++) {
            fft_in[n][0] = signal[start + n] * window[n];
            fft_in[n][1] = 0.0;
        }

        /* Execute FFT */
        fftw_execute(plan);

        /* Accumulate power spectrum */
        for (size_t k = 0; k < num_bins; k++) {
            double re = fft_out[k][0];
            double im = fft_out[k][1];
            psd_accum[k] += (re * re + im * im);
        }
        num_windows++;
    }

    /* Average and convert to dB */
    double window_power = 0.0;
    for (size_t n = 0; n < fft_size; n++) {
        window_power += window[n] * window[n];
    }

    /* Guard against division by zero if signal_len < fft_size */
    if (num_windows == 0 || window_power < 1e-20) {
        /* No valid windows processed, leave psd_db at initial -200 values */
        free(psd_accum);
        free(window);
        fftw_destroy_plan(plan);
        fftw_free(fft_in);
        fftw_free(fft_out);
        return;
    }

    for (size_t k = 0; k < num_bins; k++) {
        double power = psd_accum[k] / (num_windows * fft_size * window_power);
        if (power > 1e-20) {
            psd_db[k] = 10.0 * log10(power);
        } else {
            psd_db[k] = -200.0;
        }
    }

    free(psd_accum);
    free(window);
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

/* Measure band energy in dB (average-based, good for broadband signals) */
double measure_band_energy(const double *psd_db, const double *freqs,
                           size_t num_bins, double f_low, double f_high) {
    double total_power = 0.0;
    int count = 0;

    for (size_t k = 0; k < num_bins; k++) {
        if (freqs[k] >= f_low && freqs[k] < f_high) {
            /* Convert from dB back to linear, accumulate */
            total_power += pow(10.0, psd_db[k] / 10.0);
            count++;
        }
    }

    if (count > 0) {
        return 10.0 * log10(total_power / count);  /* Average power in dB */
    }
    return -200.0;
}

/* Measure peak energy in dB (good for discrete tones) */
double measure_peak_energy(const double *psd_db, const double *freqs,
                           size_t num_bins, double f_low, double f_high) {
    double peak = -200.0;

    for (size_t k = 0; k < num_bins; k++) {
        if (freqs[k] >= f_low && freqs[k] < f_high) {
            if (psd_db[k] > peak) {
                peak = psd_db[k];
            }
        }
    }

    return peak;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_rate> <output_rate> <signal_type>\n", argv[0]);
        fprintf(stderr, "Signal types: noise, multitone, sweep, alias_tones\n");
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr, "  Upsampling:   %s 48000 96000 noise\n", argv[0]);
        fprintf(stderr, "  Downsampling: %s 48000 32000 alias_tones  (BirdNET/Perch)\n", argv[0]);
        return 1;
    }

    /* Parse arguments */
    double input_rate = atof(argv[1]);
    double output_rate = atof(argv[2]);
    char *signal_name = argv[3];

    /* Determine signal type */
    signal_type_t signal_type;
    if (strcmp(signal_name, "noise") == 0) {
        signal_type = SIGNAL_NOISE;
    } else if (strcmp(signal_name, "multitone") == 0) {
        signal_type = SIGNAL_MULTITONE;
    } else if (strcmp(signal_name, "sweep") == 0) {
        signal_type = SIGNAL_SWEEP;
    } else if (strcmp(signal_name, "alias_tones") == 0) {
        signal_type = SIGNAL_ALIAS_TONES;
    } else {
        fprintf(stderr, "Unknown signal type: %s\n", signal_name);
        return 1;
    }

    /* Generate input signal */
    double *input = malloc(INPUT_SAMPLES * sizeof(double));
    if (!input) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    generate_signal(input, INPUT_SAMPLES, signal_type, input_rate);

    /* Calculate expected output samples */
    size_t output_samples = (size_t)(INPUT_SAMPLES * output_rate / input_rate + 1000);
    double *output = malloc(output_samples * sizeof(double));
    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        return 1;
    }

    /* Configure soxr for VHQ (very high quality) */
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
    soxr_quality_spec_t quality_spec = soxr_quality_spec(SOXR_VHQ, 0);

    /* Create soxr instance */
    soxr_error_t error;
    soxr_t soxr = soxr_create(input_rate, output_rate, 1, &error,
                               &io_spec, &quality_spec, NULL);

    if (error) {
        fprintf(stderr, "soxr_create error: %s\n", soxr_strerror(error));
        free(input);
        free(output);
        return 1;
    }

    /* Process the signal */
    size_t input_done, output_done, total_output = 0;
    error = soxr_process(soxr, input, INPUT_SAMPLES, &input_done,
                         output, output_samples, &output_done);

    if (error) {
        fprintf(stderr, "soxr_process error: %s\n", soxr_strerror(error));
        soxr_delete(soxr);
        free(input);
        free(output);
        return 1;
    }
    total_output = output_done;

    /* Flush remaining samples */
    size_t flush_done;
    error = soxr_process(soxr, NULL, 0, NULL,
                         output + total_output, output_samples - total_output, &flush_done);
    if (!error) {
        total_output += flush_done;
    }

    /* Compute PSD of input signal */
    size_t num_bins_in = FFT_SIZE / 2 + 1;
    double *psd_in = malloc(num_bins_in * sizeof(double));
    double *freqs_in = malloc(num_bins_in * sizeof(double));
    compute_psd(input, INPUT_SAMPLES, input_rate, psd_in, freqs_in, FFT_SIZE);

    /* Compute PSD of output signal */
    size_t num_bins_out = FFT_SIZE / 2 + 1;
    double *psd_out = malloc(num_bins_out * sizeof(double));
    double *freqs_out = malloc(num_bins_out * sizeof(double));
    compute_psd(output, total_output, output_rate, psd_out, freqs_out, FFT_SIZE);

    /* Define frequency bands based on upsampling vs downsampling */
    double orig_nyquist = input_rate / 2.0;
    double new_nyquist = output_rate / 2.0;
    int is_downsampling = (output_rate < input_rate);

    double passband_end, stopband_start, stopband_end;
    double passband_energy_in, passband_energy_out;
    double stopband_energy_in, stopband_energy_out;
    double attenuation;

    if (is_downsampling) {
        /*
         * DOWNSAMPLING (anti-aliasing)
         * For 48kHz -> 32kHz:
         *   - Output Nyquist: 16kHz (what we preserve)
         *   - Passband: 0 to 90% of output Nyquist (0-14.4kHz)
         *   - Aliasing region: output Nyquist to input Nyquist (16-24kHz)
         *   - These frequencies alias to (0 to input_nyquist - output_nyquist)
         */
        passband_end = new_nyquist * 0.9;
        stopband_start = new_nyquist + 500;  /* Start of aliasing region in input */
        stopband_end = orig_nyquist - 500;   /* End of aliasing region in input */

        if (signal_type == SIGNAL_ALIAS_TONES) {
            /*
             * For alias_tones signal: tones are ONLY in the aliasing region.
             * Use peak-based measurement.
             *
             * Input: peaks at 17, 18, 19, 20, 21, 22, 23 kHz (aliasing region)
             * After downsampling, these would alias to:
             *   17 kHz -> 17 - 16 = 1 kHz (or 16*2 - 17 = 15 kHz, folded)
             *   Actually for 48->32: f' = |f - 2*new_nyquist| = |f - 32kHz|
             *   17 kHz -> |17 - 32| = 15 kHz... but 15 < 16, so that's in passband
             *   Wait, aliasing formula: f_alias = |f mod (2*Fs_out) - Fs_out|
             *   For Fs_out = 32kHz: f_alias = |f mod 32 - 16|
             *   17 kHz: |17 mod 32 - 16| = |17 - 16| = 1 kHz
             *   20 kHz: |20 - 16| = 4 kHz
             *   23 kHz: |23 - 16| = 7 kHz
             *
             * So we measure: input peak in aliasing region vs output peak in alias target
             */
            stopband_energy_in = measure_peak_energy(psd_in, freqs_in, num_bins_in,
                                                      stopband_start, stopband_end);

            /* Alias target region: 0 to (orig_nyquist - new_nyquist) */
            double alias_region_end = orig_nyquist - new_nyquist;
            stopband_energy_out = measure_peak_energy(psd_out, freqs_out, num_bins_out,
                                                       100, alias_region_end);

            /* Passband is empty for alias_tones */
            passband_energy_in = -200.0;
            passband_energy_out = -200.0;

            /* Attenuation = input peak - output peak (should be very high) */
            attenuation = stopband_energy_in - stopband_energy_out;

            printf("# Anti-aliasing Test Results (DOWNSAMPLING, alias_tones)\n");
            printf("# ======================================================\n");
            printf("# Input rate:  %.0f Hz\n", input_rate);
            printf("# Output rate: %.0f Hz\n", output_rate);
            printf("# Signal type: %s\n", signal_name);
            printf("# Input samples: %d\n", INPUT_SAMPLES);
            printf("# Output samples: %zu\n", total_output);
            printf("#\n");
            printf("# Test methodology:\n");
            printf("#   - Input contains tones ONLY in aliasing region (%.0f-%.0f Hz)\n",
                   stopband_start, stopband_end);
            printf("#   - After proper anti-aliasing, output should be near silent\n");
            printf("#   - Aliased tones would appear at 0-%.0f Hz\n", alias_region_end);
            printf("#\n");
            printf("# Peak energy measurements (dB):\n");
            printf("#   Input aliasing region peak:  %.2f dB\n", stopband_energy_in);
            printf("#   Output alias target peak:    %.2f dB\n", stopband_energy_out);
            printf("#\n");
            printf("# ANTI-ALIASING ATTENUATION: %.2f dB\n", attenuation);
        } else {
            /* Broadband signal: use average-based measurement */
            passband_energy_in = measure_band_energy(psd_in, freqs_in, num_bins_in,
                                                      100, passband_end);
            passband_energy_out = measure_band_energy(psd_out, freqs_out, num_bins_out,
                                                       100, passband_end);

            stopband_energy_in = measure_band_energy(psd_in, freqs_in, num_bins_in,
                                                      stopband_start, stopband_end);

            double alias_region_end = orig_nyquist - new_nyquist;
            stopband_energy_out = measure_band_energy(psd_out, freqs_out, num_bins_out,
                                                       100, alias_region_end);

            /*
             * For broadband noise, passband contains legitimate signal.
             * Better metric: compare passband energy to output noise floor.
             */
            attenuation = passband_energy_out - stopband_energy_out;

            printf("# Anti-aliasing Test Results (DOWNSAMPLING, broadband)\n");
            printf("# ====================================================\n");
            printf("# Input rate:  %.0f Hz\n", input_rate);
            printf("# Output rate: %.0f Hz\n", output_rate);
            printf("# Signal type: %s\n", signal_name);
            printf("# Input samples: %d\n", INPUT_SAMPLES);
            printf("# Output samples: %zu\n", total_output);
            printf("#\n");
            printf("# Note: For broadband signals, use 'alias_tones' for accurate measurement\n");
            printf("#\n");
            printf("# Energy measurements (dB):\n");
            printf("#   Input passband energy:     %.2f dB\n", passband_energy_in);
            printf("#   Output passband energy:    %.2f dB\n", passband_energy_out);
            printf("#   Input aliasing region:     %.2f dB\n", stopband_energy_in);
            printf("#   Output alias target region: %.2f dB\n", stopband_energy_out);
            printf("#\n");
            printf("# PASSBAND vs ALIAS_TARGET: %.2f dB\n", attenuation);
        }
    } else {
        /*
         * UPSAMPLING (anti-imaging)
         * Original logic: measure imaging region in output
         */
        passband_end = orig_nyquist * 0.9;
        stopband_start = orig_nyquist + 1000;
        stopband_end = new_nyquist - 1000;

        /* Measure energy in each band */
        passband_energy_in = measure_band_energy(psd_in, freqs_in, num_bins_in,
                                                  100, passband_end);
        passband_energy_out = measure_band_energy(psd_out, freqs_out, num_bins_out,
                                                   100, passband_end);

        /* Stopband energy should be measured in output only */
        stopband_energy_out = measure_band_energy(psd_out, freqs_out, num_bins_out,
                                                   stopband_start, stopband_end);

        attenuation = passband_energy_out - stopband_energy_out;

        /* Print results */
        printf("# Anti-imaging Test Results (UPSAMPLING)\n");
        printf("# ======================================\n");
        printf("# Input rate:  %.0f Hz\n", input_rate);
        printf("# Output rate: %.0f Hz\n", output_rate);
        printf("# Signal type: %s\n", signal_name);
        printf("# Input samples: %d\n", INPUT_SAMPLES);
        printf("# Output samples: %zu\n", total_output);
        printf("#\n");
        printf("# Frequency bands:\n");
        printf("#   Passband:   0 - %.0f Hz\n", passband_end);
        printf("#   Stopband:   %.0f - %.0f Hz (imaging region)\n", stopband_start, stopband_end);
        printf("#\n");
        printf("# Energy measurements (dB):\n");
        printf("#   Input passband energy:   %.2f dB\n", passband_energy_in);
        printf("#   Output passband energy:  %.2f dB\n", passband_energy_out);
        printf("#   Output stopband energy:  %.2f dB\n", stopband_energy_out);
        printf("#\n");
        printf("# ANTI-IMAGING ATTENUATION: %.2f dB\n", attenuation);
    }

    printf("# STOPBAND ATTENUATION: %.2f dB\n", attenuation);
    printf("#\n");

    /* Output PSD data in CSV format for plotting */
    printf("# Output PSD data (frequency_hz, power_db):\n");
    printf("OUTPUT_PSD_START\n");
    for (size_t k = 1; k < num_bins_out; k++) {  /* Skip DC */
        printf("%.2f,%.2f\n", freqs_out[k], psd_out[k]);
    }
    printf("OUTPUT_PSD_END\n");

    /* Cleanup */
    soxr_delete(soxr);
    free(input);
    free(output);
    free(psd_in);
    free(freqs_in);
    free(psd_out);
    free(freqs_out);

    return 0;
}
