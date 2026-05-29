package resampler

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

type processIntoResampler interface {
	ProcessInto(input, output []float64) (int, error)
	EstimateOutput(inputLen int) int
}

func makeDeterministicInput(rng *rand.Rand, inputLen int, inputRate float64) []float64 {
	input := make([]float64, inputLen)
	phase1 := rng.Float64() * 2 * math.Pi
	phase2 := rng.Float64() * 2 * math.Pi
	for i := range input {
		t := float64(i) / inputRate
		input[i] = 0.7*math.Sin(2*math.Pi*440*t+phase1) +
			0.2*math.Sin(2*math.Pi*1750*t+phase2) +
			0.1*(rng.Float64()-0.5)
	}
	return input
}

// TestProcessInto_MatchesProcess is a permanent regression test verifying that
// ProcessInto produces bit-identical output to Process for every common rate
// pair. This guards against future refactors accidentally diverging the two
// code paths (analogous to buffer_integrity_test.go for the aliasing contract).
func TestProcessInto_MatchesProcess(t *testing.T) {
	ratePairs := []struct {
		inRate, outRate float64
		durSeconds      int
	}{
		{48000, 16000, 3}, // birdnet-go v2.4 primary (3:1 downsample)
		{48000, 32000, 3}, // birdnet-go v3.0 (3:2 downsample, 3s)
		{48000, 32000, 5}, // birdnet-go v3.0 production clip size
		{44100, 48000, 3}, // CD to DAT
		{48000, 44100, 3}, // DAT to CD
		{96000, 48000, 3}, // 2:1 downsample
		{48000, 96000, 3}, // 1:2 upsample
		{22050, 16000, 3}, // speech downconversion
		{16000, 48000, 3}, // speech to standard
		{44100, 32000, 5}, // CD capture to v3.0 model
	}

	qualities := []QualityPreset{
		QualityLow,
		QualityMedium,
		QualityHigh,
	}

	for _, rp := range ratePairs {
		for _, q := range qualities {
			name := fmt.Sprintf("%gto%g_%ds_q%d", rp.inRate, rp.outRate, rp.durSeconds, q)
			t.Run(name, func(t *testing.T) {
				rProcess, err := NewEngine(rp.inRate, rp.outRate, q)
				if err != nil {
					t.Fatal(err)
				}
				rInto, err := NewEngine(rp.inRate, rp.outRate, q)
				if err != nil {
					t.Fatal(err)
				}

				inputLen := int(rp.inRate) * rp.durSeconds
				input := make([]float64, inputLen)
				for i := range input {
					phase := 2.0 * math.Pi * float64(i) * 440.0 / rp.inRate
					input[i] = math.Sin(phase + 0.1*math.Sin(2.0*math.Pi*float64(i)/float64(inputLen)))
				}

				outProcess, err := rProcess.Process(input)
				if err != nil {
					t.Fatal(err)
				}

				outBuf := make([]float64, rInto.EstimateOutput(len(input)))
				n, err := rInto.ProcessInto(input, outBuf)
				if err != nil {
					t.Fatal(err)
				}
				outInto := outBuf[:n]

				if len(outProcess) != len(outInto) {
					t.Fatalf("length mismatch: Process=%d, ProcessInto=%d",
						len(outProcess), len(outInto))
				}

				for i := range outProcess {
					if outProcess[i] != outInto[i] {
						t.Fatalf("sample %d differs: Process=%v, ProcessInto=%v",
							i, outProcess[i], outInto[i])
					}
				}
			})
		}
	}
}

// TestProcessInto_ZeroAllocs enforces the zero-allocation invariant as a hard
// test failure. Uses testing.AllocsPerRun to measure steady-state allocations
// after internal buffers have grown to their final size.
func TestProcessInto_ZeroAllocs(t *testing.T) { //nolint:dupl // intentional parallel structure with TestProcessIntoFloat32_ZeroAllocs
	cases := []struct {
		name            string
		inRate, outRate float64
		durSeconds      int
	}{
		{"48to16_3s", 48000, 16000, 3},
		{"48to32_3s", 48000, 32000, 3},
		{"48to32_5s", 48000, 32000, 5},
		{"44100to32_5s", 44100, 32000, 5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := NewEngine(tc.inRate, tc.outRate, QualityMedium)
			if err != nil {
				t.Fatal(err)
			}

			inputLen := int(tc.inRate) * tc.durSeconds
			input := make([]float64, inputLen)
			for i := range input {
				input[i] = float64(i) * 1e-5
			}
			output := make([]float64, r.EstimateOutput(inputLen))

			// Warm up: let internal buffers grow to steady-state size.
			r.Reset()
			if _, err := r.ProcessInto(input, output); err != nil {
				t.Fatal(err)
			}

			allocs := testing.AllocsPerRun(100, func() {
				r.Reset()
				_, _ = r.ProcessInto(input, output)
			})

			if allocs != 0 {
				t.Fatalf("ProcessInto allocated %.0f times per call; expected 0", allocs)
			}
		})
	}
}

// TestProcessInto_BufferTooSmall verifies that ErrBufferTooSmall is returned
// when the output buffer is insufficient.
func TestProcessInto_BufferTooSmall(t *testing.T) {
	r, err := NewEngine(48000, 96000, QualityMedium) // 2x upsample
	if err != nil {
		t.Fatal(err)
	}

	input := make([]float64, 48000)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}

	tinyOutput := make([]float64, 10) // way too small
	_, err = r.ProcessInto(input, tinyOutput)
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("expected ErrBufferTooSmall, got %v", err)
	}
}

// TestProcessInto_BufferTooSmallDoesNotAdvanceState verifies callers can retry
// with a larger buffer without losing output from the failed attempt.
func TestProcessInto_BufferTooSmallDoesNotAdvanceState(t *testing.T) {
	const (
		inRate  = 48000.0
		outRate = 32000.0
	)

	input := make([]float64, int(inRate*3))
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 440.0 * float64(i) / inRate)
	}

	expectedResampler, err := NewEngine(inRate, outRate, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}
	expectedBuf := make([]float64, expectedResampler.EstimateOutput(len(input)))
	expectedN, err := expectedResampler.ProcessInto(input, expectedBuf)
	if err != nil {
		t.Fatal(err)
	}
	expected := expectedBuf[:expectedN]

	retryResampler, err := NewEngine(inRate, outRate, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}

	tinyBuf := make([]float64, 1)
	_, err = retryResampler.ProcessInto(input, tinyBuf)
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("expected ErrBufferTooSmall on first attempt, got %v", err)
	}

	retryBuf := make([]float64, retryResampler.EstimateOutput(len(input)))
	retryN, err := retryResampler.ProcessInto(input, retryBuf)
	if err != nil {
		t.Fatalf("retry failed after ErrBufferTooSmall: %v", err)
	}
	retryOutput := retryBuf[:retryN]

	if len(retryOutput) != len(expected) {
		t.Fatalf("retry length mismatch: got %d, expected %d", len(retryOutput), len(expected))
	}
	for i := range expected {
		if retryOutput[i] != expected[i] {
			t.Fatalf("retry output mismatch at sample %d: got %v, expected %v", i, retryOutput[i], expected[i])
		}
	}
}

// TestProcessInto_BufferTooSmall_NewPath verifies the New(...) API path returns
// ErrBufferTooSmall instead of silently truncating output.
func TestProcessInto_BufferTooSmall_NewPath(t *testing.T) {
	r, err := New(&Config{
		InputRate:  48000,
		OutputRate: 96000,
		Channels:   1,
		Quality:    QualitySpec{Preset: QualityMedium},
	})
	if err != nil {
		t.Fatal(err)
	}

	processIntoR, ok := r.(processIntoResampler)
	if !ok {
		t.Fatal("New(...) result does not implement ProcessInto")
	}

	input := make([]float64, 48000)
	for i := range input {
		input[i] = float64(i) * 1e-5
	}

	tinyOutput := make([]float64, 1)
	_, err = processIntoR.ProcessInto(input, tinyOutput)
	if !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("expected ErrBufferTooSmall, got %v", err)
	}
}

// TestProcessInto_MultipleChunks verifies that ProcessInto produces identical
// output to Process when called with multiple sequential chunks (streaming).
func TestProcessInto_MultipleChunks(t *testing.T) {
	rProcess, err := NewEngine(48000, 16000, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}
	rInto, err := NewEngine(48000, 16000, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}

	chunkSize := 4800 // 100ms chunks
	numChunks := 30   // 3 seconds total

	estimatedTotal := rInto.EstimateOutput(chunkSize) * numChunks
	allProcess := make([]float64, 0, estimatedTotal)
	allInto := make([]float64, 0, estimatedTotal)
	outBuf := make([]float64, rInto.EstimateOutput(chunkSize))

	for c := range numChunks {
		chunk := make([]float64, chunkSize)
		for i := range chunk {
			sample := c*chunkSize + i
			chunk[i] = math.Sin(2.0 * math.Pi * 440.0 * float64(sample) / 48000.0)
		}

		out, err := rProcess.Process(chunk)
		if err != nil {
			t.Fatal(err)
		}
		allProcess = append(allProcess, out...)

		n, err := rInto.ProcessInto(chunk, outBuf)
		if err != nil {
			t.Fatal(err)
		}
		allInto = append(allInto, outBuf[:n]...)
	}

	if len(allProcess) != len(allInto) {
		t.Fatalf("streaming length mismatch: Process=%d, ProcessInto=%d",
			len(allProcess), len(allInto))
	}

	for i := range allProcess {
		if allProcess[i] != allInto[i] {
			t.Fatalf("streaming sample %d differs: Process=%v, ProcessInto=%v",
				i, allProcess[i], allInto[i])
		}
	}
}

// TestEstimateOutput_UpperBound_NewEngine verifies that EstimateOutput provides
// a sufficient upper bound for ProcessInto during streaming workloads.
func TestEstimateOutput_UpperBound_NewEngine(t *testing.T) {
	type rateCase struct {
		name            string
		inRate, outRate float64
	}

	cases := []rateCase{
		{"48to16", 48000, 16000},
		{"48to32", 48000, 32000},
		{"44_1to32", 44100, 32000},
		{"44_1to48", 44100, 48000},
		{"48to44_1", 48000, 44100},
	}

	const (
		maxChunkLen = 24000
		iterations  = 120
	)

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(1337 + int64(idx)))

			rProcess, err := NewEngine(tc.inRate, tc.outRate, QualityMedium)
			if err != nil {
				t.Fatal(err)
			}
			rInto, err := NewEngine(tc.inRate, tc.outRate, QualityMedium)
			if err != nil {
				t.Fatal(err)
			}

			for i := range iterations {
				inputLen := 1 + rng.Intn(maxChunkLen)
				input := makeDeterministicInput(rng, inputLen, tc.inRate)

				expected, err := rProcess.Process(input)
				if err != nil {
					t.Fatalf("Process failed at iteration %d: %v", i, err)
				}

				estimate := rInto.EstimateOutput(inputLen)
				output := make([]float64, estimate)
				n, err := rInto.ProcessInto(input, output)
				if err != nil {
					t.Fatalf("ProcessInto failed at iteration %d (input=%d, estimate=%d): %v", i, inputLen, estimate, err)
				}
				if n > estimate {
					t.Fatalf("ProcessInto wrote beyond estimate at iteration %d: wrote=%d estimate=%d", i, n, estimate)
				}
				if n != len(expected) {
					t.Fatalf("length mismatch at iteration %d: Process=%d ProcessInto=%d", i, len(expected), n)
				}
				for sampleIdx := range expected {
					if output[sampleIdx] != expected[sampleIdx] {
						t.Fatalf(
							"sample mismatch at iteration %d sample %d: Process=%v ProcessInto=%v",
							i,
							sampleIdx,
							expected[sampleIdx],
							output[sampleIdx],
						)
					}
				}
			}
		})
	}
}

// TestEstimateOutput_UpperBound_NewPath verifies that EstimateOutput is
// sufficient for ProcessInto on the New(...) API path during streaming.
func TestEstimateOutput_UpperBound_NewPath(t *testing.T) {
	type rateCase struct {
		name            string
		inRate, outRate float64
	}

	cases := []rateCase{
		{"48to16", 48000, 16000},
		{"48to32", 48000, 32000},
		{"44_1to32", 44100, 32000},
		{"44_1to48", 44100, 48000},
		{"48to44_1", 48000, 44100},
	}

	const (
		maxChunkLen = 24000
		iterations  = 80
	)

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(7331 + int64(idx)))

			cfg := &Config{
				InputRate:  tc.inRate,
				OutputRate: tc.outRate,
				Channels:   1,
				Quality:    QualitySpec{Preset: QualityMedium},
			}

			rProcess, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			rIntoRaw, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			rInto, ok := rIntoRaw.(processIntoResampler)
			if !ok {
				t.Fatal("New(...) result does not implement ProcessInto")
			}

			for i := range iterations {
				inputLen := 1 + rng.Intn(maxChunkLen)
				input := makeDeterministicInput(rng, inputLen, tc.inRate)

				expected, err := rProcess.Process(input)
				if err != nil {
					t.Fatalf("Process failed at iteration %d: %v", i, err)
				}

				estimate := rInto.EstimateOutput(inputLen)
				output := make([]float64, estimate)
				n, err := rInto.ProcessInto(input, output)
				if err != nil {
					t.Fatalf("ProcessInto failed at iteration %d (input=%d, estimate=%d): %v", i, inputLen, estimate, err)
				}
				if n > estimate {
					t.Fatalf("ProcessInto wrote beyond estimate at iteration %d: wrote=%d estimate=%d", i, n, estimate)
				}
				if n != len(expected) {
					t.Fatalf("length mismatch at iteration %d: Process=%d ProcessInto=%d", i, len(expected), n)
				}
			}
		})
	}
}

// TestProcessIntoFloat32_ZeroAllocs enforces the zero-allocation invariant for
// the float32 engine path (SimpleResamplerFloat32.ProcessInto), mirroring
// TestProcessInto_ZeroAllocs. The engine is float32-native, so there is no
// float64 round-trip on this path.
func TestProcessIntoFloat32_ZeroAllocs(t *testing.T) { //nolint:dupl // intentional float32 mirror of TestProcessInto_ZeroAllocs
	cases := []struct {
		name            string
		inRate, outRate float64
		durSeconds      int
	}{
		{"48to16_3s", 48000, 16000, 3},
		{"48to32_3s", 48000, 32000, 3},
		{"48to32_5s", 48000, 32000, 5},
		{"44100to32_5s", 44100, 32000, 5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := NewEngineFloat32(tc.inRate, tc.outRate, QualityMedium)
			if err != nil {
				t.Fatal(err)
			}

			inputLen := int(tc.inRate) * tc.durSeconds
			input := make([]float32, inputLen)
			for i := range input {
				input[i] = float32(i) * 1e-5
			}
			output := make([]float32, r.EstimateOutput(inputLen))

			// Warm up: let internal buffers grow to steady-state size.
			r.Reset()
			if _, err := r.ProcessInto(input, output); err != nil {
				t.Fatal(err)
			}

			allocs := testing.AllocsPerRun(100, func() {
				r.Reset()
				_, _ = r.ProcessInto(input, output)
			})

			if allocs != 0 {
				t.Fatalf("ProcessInto allocated %.0f times per call; expected 0", allocs)
			}
		})
	}
}

// TestProcessFloat32Into_ZeroAllocs enforces the zero-allocation invariant for
// the float32 New(...) batch path (constantRateResampler.ProcessFloat32Into).
// This path converts float32<->float64 through grow-only scratch buffers, so it
// is exercised as continuous warm streaming (no Reset between iterations) to
// match the steady-state usage the scratch buffers are designed for.
func TestProcessFloat32Into_ZeroAllocs(t *testing.T) {
	cases := []struct {
		name            string
		inRate, outRate float64
		durSeconds      int
	}{
		{"48to16_3s", 48000, 16000, 3},
		{"48to32_3s", 48000, 32000, 3},
		{"48to32_5s", 48000, 32000, 5},
		{"44100to32_5s", 44100, 32000, 5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := New(&Config{
				InputRate:  tc.inRate,
				OutputRate: tc.outRate,
				Channels:   1,
				Quality:    QualitySpec{Preset: QualityMedium},
			})
			if err != nil {
				t.Fatal(err)
			}
			r, ok := raw.(processFloat32IntoResampler)
			if !ok {
				t.Fatal("New(...) result does not implement ProcessFloat32Into")
			}

			inputLen := int(tc.inRate) * tc.durSeconds
			input := make([]float32, inputLen)
			for i := range input {
				input[i] = float32(i) * 1e-5
			}
			output := make([]float32, r.EstimateOutput(inputLen))

			// Warm up: let internal ring buffers and scratch slices grow to
			// steady-state size.
			if _, err := r.ProcessFloat32Into(input, output); err != nil {
				t.Fatal(err)
			}

			allocs := testing.AllocsPerRun(100, func() {
				_, _ = r.ProcessFloat32Into(input, output)
			})

			if allocs != 0 {
				t.Fatalf("ProcessFloat32Into allocated %.0f times per call; expected 0", allocs)
			}
		})
	}
}

// TestNewPath_ProcessInto_MatchesProcess is a permanent regression test verifying
// that on the New(config) path (constantRateResampler), ProcessInto produces
// bit-identical output to Process across common rate pairs and qualities. Process
// drives processChannel and ProcessInto drives processChannelInto; the two read
// the staged input buffers with different granularity but must agree
// sample-for-sample.
func TestNewPath_ProcessInto_MatchesProcess(t *testing.T) {
	ratePairs := []struct {
		inRate, outRate float64
		durSeconds      int
	}{
		{48000, 16000, 3},
		{48000, 32000, 3},
		{44100, 48000, 3},
		{48000, 44100, 3},
		{96000, 48000, 3},
		{16000, 48000, 3},
		{22050, 16000, 3},
	}
	qualities := []QualityPreset{QualityLow, QualityMedium, QualityHigh}

	for idx, rp := range ratePairs {
		for _, q := range qualities {
			t.Run(fmt.Sprintf("%gto%g_q%d", rp.inRate, rp.outRate, q), func(t *testing.T) {
				cfg := &Config{
					InputRate:  rp.inRate,
					OutputRate: rp.outRate,
					Channels:   1,
					Quality:    QualitySpec{Preset: q},
				}

				rProcess, err := New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				rIntoRaw, err := New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				rInto, ok := rIntoRaw.(processIntoResampler)
				if !ok {
					t.Fatal("New(...) result does not implement ProcessInto")
				}

				rng := rand.New(rand.NewSource(4242 + int64(idx)))
				input := makeDeterministicInput(rng, int(rp.inRate)*rp.durSeconds, rp.inRate)

				outProcess, err := rProcess.Process(input)
				if err != nil {
					t.Fatal(err)
				}

				outBuf := make([]float64, rInto.EstimateOutput(len(input)))
				n, err := rInto.ProcessInto(input, outBuf)
				if err != nil {
					t.Fatal(err)
				}

				assertSamplesEqual(t, "process-vs-into", outBuf[:n], outProcess)
			})
		}
	}
}
