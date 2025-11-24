CC = gcc
CFLAGS = -Wall -Wextra -O2
CFLAGS_BENCH = -Wall -Wextra -O3
LDFLAGS = -lsoxr -lm
LDFLAGS_FFT = -lsoxr -lm -lfftw3

TARGET = test_soxr_reference
BENCH_TARGET = bench_soxr
ANTIALIAS_TARGET = test_antialiasing
QUALITY_TARGET = test_quality
SRC = test_soxr_reference.c
BENCH_SRC = bench_soxr.c
ANTIALIAS_SRC = test_antialiasing.c
QUALITY_SRC = test_quality.c

all: $(TARGET) $(BENCH_TARGET) $(ANTIALIAS_TARGET) $(QUALITY_TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

$(BENCH_TARGET): $(BENCH_SRC)
	$(CC) $(CFLAGS_BENCH) -o $(BENCH_TARGET) $(BENCH_SRC) $(LDFLAGS)

$(ANTIALIAS_TARGET): $(ANTIALIAS_SRC)
	$(CC) $(CFLAGS) -o $(ANTIALIAS_TARGET) $(ANTIALIAS_SRC) $(LDFLAGS_FFT)

$(QUALITY_TARGET): $(QUALITY_SRC)
	$(CC) $(CFLAGS) -o $(QUALITY_TARGET) $(QUALITY_SRC) $(LDFLAGS_FFT)

clean:
	rm -f $(TARGET) $(BENCH_TARGET) $(ANTIALIAS_TARGET) $(QUALITY_TARGET)

test: $(TARGET)
	@echo "Testing 2x upsampling..."
	@./$(TARGET) 44100 88200 dc 2>/dev/null | head -20
	@echo ""
	@echo "Testing CD to DAT..."
	@./$(TARGET) 44100 48000 dc 2>/dev/null | head -20

test-antialias: $(ANTIALIAS_TARGET)
	@echo "=== Testing upsampling (anti-imaging) ==="
	@./$(ANTIALIAS_TARGET) 48000 96000 noise 2>/dev/null | grep -E "^#"
	@echo ""
	@echo "=== Testing downsampling 48kHz->32kHz (anti-aliasing, BirdNET/Perch) ==="
	@./$(ANTIALIAS_TARGET) 48000 32000 noise 2>/dev/null | grep -E "^#"
	@echo ""
	@echo "=== Testing downsampling 48kHz->44.1kHz (DAT to CD) ==="
	@./$(ANTIALIAS_TARGET) 48000 44100 noise 2>/dev/null | grep -E "^#"

bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)

test-quality: $(QUALITY_TARGET)
	@echo "=== Passband Ripple Test ==="
	@./$(QUALITY_TARGET) 44100 48000 ripple 2>/dev/null | grep -E "^#"
	@echo ""
	@echo "=== THD Test @ 1kHz ==="
	@./$(QUALITY_TARGET) 44100 48000 thd:1000 2>/dev/null | grep -E "^#"
	@echo ""
	@echo "=== SNR Test @ 1kHz ==="
	@./$(QUALITY_TARGET) 44100 48000 snr:1000 2>/dev/null | grep -E "^#"
	@echo ""
	@echo "=== Impulse Response Test ==="
	@./$(QUALITY_TARGET) 44100 48000 impulse 2>/dev/null | grep -E "^#"

.PHONY: all clean test test-antialias test-quality bench
