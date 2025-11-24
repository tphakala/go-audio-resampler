/*
 * soxr benchmark for comparing with Go implementation
 *
 * Compiles with: gcc -O3 -o bench_soxr bench_soxr.c -lsoxr -lm
 * Usage: ./bench_soxr
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <soxr.h>

#define M_PI 3.14159265358979323846

/* Benchmark configuration */
typedef struct {
    const char *name;
    double input_rate;
    double output_rate;
    size_t input_samples;
    int iterations;
} benchmark_config_t;

/* Generate sine wave test signal */
void generate_sine(double *buffer, size_t num_samples, double sample_rate, double frequency) {
    for (size_t i = 0; i < num_samples; i++) {
        double phase = 2.0 * M_PI * frequency * (double)i / sample_rate;
        buffer[i] = sin(phase);
    }
}

/* Get current time in nanoseconds */
long long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Run a single benchmark */
void run_benchmark(benchmark_config_t *config) {
    /* Allocate input buffer */
    double *input = malloc(config->input_samples * sizeof(double));
    if (!input) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    generate_sine(input, config->input_samples, config->input_rate, 1000.0);

    /* Calculate output buffer size */
    size_t output_samples = (size_t)(config->input_samples * config->output_rate / config->input_rate + 1000);
    double *output = malloc(output_samples * sizeof(double));
    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        return;
    }

    /* Configure soxr for high quality */
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
    soxr_quality_spec_t quality_spec = soxr_quality_spec(SOXR_VHQ, 0);

    /* Create soxr instance */
    soxr_error_t error;
    soxr_t soxr = soxr_create(config->input_rate, config->output_rate, 1, &error,
                               &io_spec, &quality_spec, NULL);
    if (error) {
        fprintf(stderr, "soxr_create error: %s\n", soxr_strerror(error));
        free(input);
        free(output);
        return;
    }

    /* Warm up */
    size_t input_done, output_done;
    soxr_process(soxr, input, config->input_samples, &input_done,
                 output, output_samples, &output_done);
    soxr_clear(soxr);

    /* Benchmark loop */
    long long start_time = get_time_ns();
    size_t total_samples_out = 0;

    for (int i = 0; i < config->iterations; i++) {
        soxr_process(soxr, input, config->input_samples, &input_done,
                     output, output_samples, &output_done);
        total_samples_out += output_done;
        soxr_clear(soxr);
    }

    long long end_time = get_time_ns();
    double elapsed_sec = (double)(end_time - start_time) / 1e9;
    double samples_per_sec = (double)(config->input_samples * config->iterations) / elapsed_sec;
    double ns_per_op = (double)(end_time - start_time) / config->iterations;

    /* Print results in a format easy to compare with Go benchmarks */
    printf("%-25s  %10d iterations  %12.0f ns/op  %12.2f samples/sec\n",
           config->name, config->iterations, ns_per_op, samples_per_sec);

    /* Cleanup */
    soxr_delete(soxr);
    free(input);
    free(output);
}

int main(void) {
    printf("soxr Benchmarks (VHQ quality)\n");
    printf("============================================================\n\n");

    /* Define benchmarks matching Go implementation */
    benchmark_config_t benchmarks[] = {
        /* Standard conversions - 1 second of audio */
        {"CD_to_DAT_1sec", 44100, 48000, 44100, 100},
        {"DAT_to_CD_1sec", 48000, 44100, 48000, 100},
        {"48k_to_32k_1sec", 48000, 32000, 48000, 100},
        {"2x_upsample_1sec", 44100, 88200, 44100, 100},

        /* Streaming chunks - 1024 samples (common buffer size) */
        {"CD_to_DAT_1024", 44100, 48000, 1024, 1000},
        {"DAT_to_CD_1024", 48000, 44100, 1024, 1000},
        {"48k_to_32k_1024", 48000, 32000, 1024, 1000},

        /* Streaming chunks - 4096 samples */
        {"CD_to_DAT_4096", 44100, 48000, 4096, 500},
        {"DAT_to_CD_4096", 48000, 44100, 4096, 500},
        {"48k_to_32k_4096", 48000, 32000, 4096, 500},
    };

    size_t num_benchmarks = sizeof(benchmarks) / sizeof(benchmarks[0]);

    for (size_t i = 0; i < num_benchmarks; i++) {
        run_benchmark(&benchmarks[i]);
    }

    printf("\n");
    return 0;
}
