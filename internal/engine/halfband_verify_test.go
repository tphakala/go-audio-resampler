// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"fmt"
	"testing"
)

func TestHalfBandDetection(t *testing.T) {
	qualities := []struct {
		name    string
		quality Quality
	}{
		{"Low", QualityLow},
		{"Medium", QualityMedium},
		{"High", QualityHigh},
	}

	fmt.Println("\n=== Half-Band Detection ===")
	for _, q := range qualities {
		stage, err := NewDFTStage[float64](2, q.quality)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("%s: isHalfBand=%v, phase0TapOffset=%d, phase0TapScale=%.6f\n",
			q.name, stage.isHalfBand, stage.phase0TapOffset, stage.phase0TapScale)
	}
}
