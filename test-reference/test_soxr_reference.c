/*
 * soxr reference implementation for testing Go audio resampler
 *
 * Compiles with: gcc -o test_soxr_reference test_soxr_reference.c -lsoxr -lm
 * Usage: ./test_soxr_reference <input_rate> <output_rate> <signal_type> [frequency]
 *
 * signal_type:
 *   dc       - DC signal (value=1.0)
 *   sine     - Sine wave at specified frequency (default 1000Hz)
 *   impulse  - Impulse signal
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <soxr.h>

#define INPUT_SAMPLES 4000
#define M_PI 3.14159265358979323846

typedef enum {
    SIGNAL_DC,
    SIGNAL_SINE,
    SIGNAL_IMPULSE
} signal_type_t;

/* Generate test signal */
void generate_signal(double *buffer, size_t num_samples, signal_type_t type,
                     double sample_rate, double frequency) {
    size_t i;

    switch (type) {
        case SIGNAL_DC:
            for (i = 0; i < num_samples; i++) {
                buffer[i] = 1.0;
            }
            break;

        case SIGNAL_SINE:
            for (i = 0; i < num_samples; i++) {
                double phase = 2.0 * M_PI * frequency * i / sample_rate;
                buffer[i] = sin(phase);
            }
            break;

        case SIGNAL_IMPULSE:
            memset(buffer, 0, num_samples * sizeof(double));
            buffer[num_samples / 2] = 1.0;  /* Impulse in the middle */
            break;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_rate> <output_rate> <signal_type> [frequency]\n", argv[0]);
        fprintf(stderr, "Signal types: dc, sine, impulse\n");
        return 1;
    }

    /* Parse arguments */
    double input_rate = atof(argv[1]);
    double output_rate = atof(argv[2]);
    char *signal_name = argv[3];
    double frequency = (argc > 4) ? atof(argv[4]) : 1000.0;

    /* Determine signal type */
    signal_type_t signal_type;
    if (strcmp(signal_name, "dc") == 0) {
        signal_type = SIGNAL_DC;
    } else if (strcmp(signal_name, "sine") == 0) {
        signal_type = SIGNAL_SINE;
    } else if (strcmp(signal_name, "impulse") == 0) {
        signal_type = SIGNAL_IMPULSE;
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
    generate_signal(input, INPUT_SAMPLES, signal_type, input_rate, frequency);

    /* Calculate expected output samples (with extra for filter latency) */
    size_t output_samples = (size_t)(INPUT_SAMPLES * output_rate / input_rate + 1000);
    double *output = malloc(output_samples * sizeof(double));
    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        return 1;
    }

    /* Configure soxr for high quality (matching our Go implementation) */
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);

    /* Use VHQ quality (very high quality) */
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
    output_done = total_output;

    /* Print results in CSV format for easy parsing */
    printf("# soxr reference output\n");
    printf("# input_rate: %.1f\n", input_rate);
    printf("# output_rate: %.1f\n", output_rate);
    printf("# signal_type: %s\n", signal_name);
    if (signal_type == SIGNAL_SINE) {
        printf("# frequency: %.1f\n", frequency);
    }
    printf("# input_samples: %zu\n", INPUT_SAMPLES);
    printf("# output_samples: %zu\n", output_done);
    printf("# ratio: %.10f\n", output_rate / input_rate);
    printf("#\n");
    printf("# Output samples:\n");

    for (size_t i = 0; i < output_done; i++) {
        printf("%.15f\n", output[i]);
    }

    /* Cleanup */
    soxr_delete(soxr);
    free(input);
    free(output);

    return 0;
}
