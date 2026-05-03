package engine

import (
	"math"
	"testing"
)

// TestStageAdapterFloat32_ProcessZeroCopy verifies the float32 adapter fallback
// path handles []float64 input without panicking.
func TestStageAdapterFloat32_ProcessZeroCopy(t *testing.T) {
	r, err := NewResampler[float32](44100, 48000, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}

	adapter := NewStageAdapter[float32](r)
	input := make([]float64, 4096)
	for i := range input {
		input[i] = math.Sin(2.0 * math.Pi * 440.0 * float64(i) / 44100.0)
	}

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("ProcessZeroCopy panicked for float32 adapter: %v", recovered)
		}
	}()

	if _, err := adapter.ProcessZeroCopy(input); err != nil {
		t.Fatalf("ProcessZeroCopy returned error: %v", err)
	}
}
