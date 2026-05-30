// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package pipeline

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildPipeline(t *testing.T) {
	tests := []struct {
		name          string
		ratio         float64
		quality       QualityParams
		expectError   bool
		expectedTypes []StageType
	}{
		{
			name:  "invalid ratio <= 0",
			ratio: 0,
			quality: QualityParams{
				Precision: 16,
			},
			expectError: true,
		},
		{
			name:  "invalid negative ratio",
			ratio: -1.5,
			quality: QualityParams{
				Precision: 16,
			},
			expectError: true,
		},
		{
			name:  "quick mode (8-bit)",
			ratio: 1.5,
			quality: QualityParams{
				Precision: 8,
			},
			expectError:   false,
			expectedTypes: []StageType{StageCubic},
		},
		{
			name:  "downsampling (ratio < 0.5)",
			ratio: 0.125, // 0.125 -> HB, HB, remainder 0.5 (StageFFT)
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageHalfBand, StageHalfBand, StageFFT},
		},
		{
			name:  "downsampling with remainder",
			ratio: 0.15, // 0.15 = 0.5 * 0.5 * 0.6 -> StageHalfBand, StageHalfBand, StagePolyphase
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageHalfBand, StageHalfBand, StagePolyphase},
		},
		{
			name:  "upsampling (ratio > 2.0)",
			ratio: 8.0, // 8.0 -> HB, HB, remainder 2.0 (StageFFT)
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageHalfBand, StageHalfBand, StageFFT},
		},
		{
			name:  "upsampling with remainder",
			ratio: 10.0, // 10.0 -> HB, HB, HB, remainder 1.25 (StagePolyphase)
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageHalfBand, StageHalfBand, StageHalfBand, StagePolyphase},
		},
		{
			name:  "common audio downsampling ratio (48000 to 44100)",
			ratio: 44100.0 / 48000.0,
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageFFT},
		},
		{
			name:  "common audio upsampling ratio (44100 to 48000)",
			ratio: 48000.0 / 44100.0,
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageFFT},
		},
		{
			name:  "high precision (>= 28-bit) uses FFT",
			ratio: 1.5,
			quality: QualityParams{
				Precision:     28,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageFFT},
		},
		{
			name:  "very high precision (32-bit) uses FFT",
			ratio: 1.5,
			quality: QualityParams{
				Precision:     32,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StageFFT},
		},
		{
			name:  "mid precision polyphase (24-bit)",
			ratio: 1.5,
			quality: QualityParams{
				Precision:     24,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{StagePolyphase},
		},
		{
			name:  "exact ratio of 1.0 (no-op)",
			ratio: 1.0,
			quality: QualityParams{
				Precision:     16,
				PassbandEnd:   0.45,
				StopbandBegin: 0.5,
			},
			expectError:   false,
			expectedTypes: []StageType{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := BuildPipeline(tt.ratio, tt.quality)
			if tt.expectError {
				require.Error(t, err)
				assert.Nil(t, p)
			} else {
				require.NoError(t, err)
				require.NotNil(t, p)
				types := []StageType{}
				for _, st := range p.GetStages() {
					types = append(types, st.Type)
				}
				assert.Equal(t, tt.expectedTypes, types)
			}
		})
	}
}

func TestPipeline_Getters(t *testing.T) {
	p := &Pipeline{
		stages: []StageSpec{
			{Type: StageCubic, Ratio: 2.0},
			{Type: StageHalfBand, Ratio: 0.5, FilterLength: 32},
			{Type: StagePolyphase, Ratio: 1.5, FilterLength: 64},
			{Type: StageFFT, Ratio: 1.2, FilterLength: 128},
			{Type: StageDelay, Ratio: 1.0, FilterLength: 10},
		},
		totalRatio: 1.5,
	}

	p.calculateLatency()

	assert.InDelta(t, 1.5, p.GetTotalRatio(), 1e-9)
	assert.NotEmpty(t, p.GetStages())
	assert.Positive(t, p.GetTotalLatency())
}

func TestOptimizePipeline(t *testing.T) {
	p := &Pipeline{
		stages: []StageSpec{
			{Type: StageCubic, Ratio: 2.0},
		},
		totalRatio: 2.0,
	}
	opt := OptimizePipeline(p)
	assert.Same(t, p, opt)
}
