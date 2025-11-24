package engine

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAnalyzeHalfbandStructure analyzes the structure of a half-band filter.
func TestAnalyzeHalfbandStructure(t *testing.T) {
	stage, err := NewDFTStage[float64](2, QualityHigh)
	require.NoError(t, err, "Failed to create DFT stage")

	t.Log("\n=== Half-band Filter Analysis ===")
	t.Logf("Factor: %d, TapsPerPhase: %d", stage.factor, stage.tapsPerPhase)

	// For a half-band filter in 2x upsampling:
	// - Phase 0: center tap = 0.5, others = 0 (ideally)
	// - Phase 1: the actual lowpass filter

	// Count near-zero coefficients
	const threshold = 1e-10
	for phase := 0; phase < stage.factor; phase++ {
		coeffs := stage.polyCoeffs[phase]
		zeroCount := 0
		nonZeroSum := 0.0
		for _, c := range coeffs {
			if c < threshold && c > -threshold {
				zeroCount++
			} else {
				nonZeroSum += c
			}
		}
		t.Logf("Phase %d: %d/%d near-zero coeffs (%.1f%%), sum=%.4f",
			phase, zeroCount, len(coeffs),
			float64(zeroCount)/float64(len(coeffs))*100, nonZeroSum)
	}

	// Check if this is a true half-band structure
	// In a perfect half-band, every other tap (except center) would be zero
	t.Log("\nSample coefficients (first 10):")
	for phase := 0; phase < stage.factor; phase++ {
		var coeffStr strings.Builder
		coeffStr.WriteString(fmt.Sprintf("Phase %d: ", phase))
		for i := 0; i < 10 && i < len(stage.polyCoeffs[phase]); i++ {
			coeffStr.WriteString(fmt.Sprintf("%.4f ", stage.polyCoeffs[phase][i]))
		}
		t.Log(coeffStr.String() + "...")
	}

	// Basic assertions to ensure the stage is valid
	assert.Equal(t, 2, stage.factor, "Expected factor of 2 for half-band")
	assert.Positive(t, stage.tapsPerPhase, "TapsPerPhase should be > 0")
	assert.Len(t, stage.polyCoeffs, stage.factor, "Should have coeffs for each phase")
}
