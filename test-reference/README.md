# SOXR Reference Implementation

This directory contains the C reference implementation using libsoxr for testing and comparison.

## Building

```bash
make
```

Or manually:
```bash
gcc -o test_soxr_reference test_soxr_reference.c -lsoxr -lm
```

## Usage

```bash
./test_soxr_reference <input_rate> <output_rate> <signal_type> [frequency [quality]]
```

### Signal Types
- `dc` - DC signal (value=1.0)
- `sine` - Sine wave at specified frequency (default 1000Hz)
- `impulse` - Impulse signal

`frequency` is ignored for `dc` and `impulse`, but must be given when `quality` is.

### Quality

`quality` selects the libsoxr recipe: `quick`, `low`, `medium`, `high` or `veryhigh` (default). An unknown name is an error. The recipe used is echoed in the `# quality:` header line, which the Go comparison tests check.

### Examples

```bash
# Test 2x upsampling with DC signal
./test_soxr_reference 44100 88200 dc

# Test CD to DAT conversion with 1kHz sine wave
./test_soxr_reference 44100 48000 sine 1000

# Same, with the libsoxr HQ recipe
./test_soxr_reference 44100 48000 sine 1000 high
```

The Go tests in `internal/engine` (`soxr_comparison_test.go`) run this tool through `task test:soxr`. That task runs `make`, which also builds `test_quality` and `test_antialiasing`, so it needs libsoxr and fftw3.

## Output Format

The program outputs resampled samples in plain text format, one sample per line, with metadata in comment lines (starting with `#`).
