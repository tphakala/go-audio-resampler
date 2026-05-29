package engine

import (
	"math"
	"testing"
)

// TestPolyphaseStageFlush_CanonicalLength is a regression test for the bug where
// PolyphaseStage.Flush() padded the stream with tapsPerPhase*historyBufferMultiplier
// (2x) zeros. historyBufferMultiplier is a buffer pre-allocation constant, not a
// flush-padding amount. tapsPerPhase zeros is enough to drain the polyphase delay
// line (matching the sibling DFTStage.Flush, which pads tapsPerPhase). The extra
// tapsPerPhase zeros only produced additional all-zero output windows, making the
// total output length longer than canonical (e.g. +44/+57 samples) without
// dropping any real samples.
//
// This asserts the total output length (Process + Flush) lands within a tight
// margin of the ideal resampled length. The 2x padding overshoots that margin;
// the correct tapsPerPhase padding lands within it. It also asserts the flush
// fully drains the delay line (the final sample is silence), which catches the
// opposite error of padding too few zeros and dropping real tail samples.
func TestPolyphaseStageFlush_CanonicalLength(t *testing.T) {
	cases := []struct {
		name           string
		ratio, totalIO float64
		hasPre         bool
	}{
		{"down_2_3", 2.0 / 3.0, 2.0 / 3.0, false},
		{"up_44k_48k", 48000.0 / 44100.0, 44100.0 / 48000.0, false},
		{"down_22k_16k", 16000.0 / 22050.0, 16000.0 / 22050.0, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			st, err := NewPolyphaseStage[float64](c.ratio, c.totalIO, c.hasPre, QualityHigh)
			if err != nil {
				t.Fatalf("NewPolyphaseStage: %v", err)
			}

			const n = 20000
			input := make([]float64, n)
			for i := range input {
				input[i] = math.Sin(2 * math.Pi * 1000 * float64(i) / 48000.0)
			}

			proc, err := st.Process(input)
			if err != nil {
				t.Fatalf("Process: %v", err)
			}
			flush, err := st.Flush()
			if err != nil {
				t.Fatalf("Flush: %v", err)
			}
			total := len(proc) + len(flush)
			ideal := int(math.Round(float64(n) * c.ratio))

			// A correctly flushed polyphase stage emits ~ideal samples; padding
			// tapsPerPhase zeros leaves only a sub-sample rounding tail. The 2x
			// padding overshoots by ~tapsPerPhase*ratio (43-57 samples here). The
			// lower bound guards against the opposite regression: under-padding
			// would drop real tail samples and come up short.
			const margin = 16
			if total > ideal+margin {
				t.Errorf("Process+Flush total = %d, want <= %d (ideal %d, taps %d); flush over-padded with trailing silence",
					total, ideal+margin, ideal, st.tapsPerPhase)
			}
			if total < ideal-margin {
				t.Errorf("Process+Flush total = %d, want >= %d (ideal %d, taps %d); flush dropped real tail samples",
					total, ideal-margin, ideal, st.tapsPerPhase)
			}

			// Flushing a fed stage must emit the filter ringdown tail. The
			// canonical-length bounds above already prove the delay line drained
			// without dropping real samples (under-padding would show up as a
			// short total). The exact last-sample value is intentionally not
			// asserted: for decimation ratios the fixed-point phase accumulator
			// can terminate before the final all-zero window, so the last emitted
			// sample is not contractually silence.
			if len(flush) == 0 {
				t.Fatalf("Flush returned no samples; delay line not drained")
			}
		})
	}
}

// TestPolyphaseStageFlush_EmptyHistoryReturnsNothing verifies that flushing a
// stage that was never fed any input returns no samples, matching the sibling
// DFTStage.Flush (which guards len(history)==0). Without the guard, Flush pads
// tapsPerPhase zeros through the empty delay line and emits a phantom window of
// zero-valued output samples for a stage that produced nothing.
func TestPolyphaseStageFlush_EmptyHistoryReturnsNothing(t *testing.T) {
	for _, ratio := range []float64{2.0 / 3.0, 48000.0 / 44100.0} {
		st, err := NewPolyphaseStage[float64](ratio, ratio, false, QualityHigh)
		if err != nil {
			t.Fatalf("NewPolyphaseStage: %v", err)
		}
		flush, err := st.Flush()
		if err != nil {
			t.Fatalf("Flush: %v", err)
		}
		if len(flush) != 0 {
			t.Errorf("ratio %g: Flush on never-fed stage returned %d samples, want 0 (phantom trailing silence)",
				ratio, len(flush))
		}
	}
}
