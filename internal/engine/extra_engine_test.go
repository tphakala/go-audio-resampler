// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCubicStageGetters(t *testing.T) {
	// CubicStage float64
	cs := NewCubicStage[float64](1.5)
	assert.InDelta(t, 1.5, cs.GetRatio(), 1e-9)
	assert.Equal(t, cubicLatencySamples, cs.GetLatency())
	assert.InDelta(t, cubicLatencySamples*1.5, cs.StartupDeficit(), 1e-9)
	assert.Equal(t, 1, cs.GetMinInput())
	assert.Equal(t, int64(cubicMemoryUsage), cs.GetMemoryUsage())
	assert.Equal(t, 4, cs.GetFilterLength()) // Hermite cubic interpolation uses 4 points
	assert.Equal(t, 0, cs.GetPhases())
	assert.Empty(t, cs.GetSIMDInfo())
}

func TestStageAdapterGetters(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	sa := NewStageAdapter(r)
	assert.InDelta(t, 48000.0/44100.0, sa.GetRatio(), 1e-9)
	assert.Positive(t, sa.GetLatency())
	assert.Equal(t, 1, sa.GetMinInput())
	assert.Positive(t, sa.GetMemoryUsage())
	assert.Positive(t, sa.GetFilterLength())
	assert.Positive(t, sa.GetPhases())
	assert.NotNil(t, sa.GetSIMDInfo())

	// DFT pre-stage but no polyphase
	r2, err := NewResampler[float64](22050, 44100, QualityHigh)
	require.NoError(t, err)
	sa2 := NewStageAdapter(r2)
	assert.Positive(t, sa2.GetLatency())
	assert.Positive(t, sa2.GetMemoryUsage())
	assert.Positive(t, sa2.GetFilterLength())
	assert.Equal(t, 0, sa2.GetPhases())
}

func TestResamplerProcessZeroCopy(t *testing.T) {
	r, err := NewResampler[float64](44100, 48000, QualityHigh)
	require.NoError(t, err)

	// Make input larger than filter latency (1000 samples)
	input := make([]float64, 1000)
	for i := range input {
		input[i] = 0.1
	}

	output, err := r.ProcessZeroCopy(input)
	require.NoError(t, err)
	assert.NotEmpty(t, output)

	// Cubic stage ProcessZeroCopy
	rc, err := NewResampler[float64](44100, 48000, QualityQuick)
	require.NoError(t, err)
	outputC, err := rc.ProcessZeroCopy(input)
	require.NoError(t, err)
	assert.NotEmpty(t, outputC)
}

func TestResamplerLargeInputs_DFTChunked(t *testing.T) {
	// Float64 upsampling to trigger DFTStage chunked path
	r64, err := NewResampler[float64](22050, 44100, QualityHigh)
	require.NoError(t, err)

	largeInput64 := make([]float64, 5000)
	for i := range largeInput64 {
		largeInput64[i] = 0.05
	}

	out64, err := r64.Process(largeInput64)
	require.NoError(t, err)
	// The stage is time-aligned, so the first call withholds only the filter's
	// group delay (331 taps: (331-1)/2 = 165 output samples): 5000 inputs * 2
	// upsampling factor - 165 = 9835 output samples
	assert.Len(t, out64, 9835)

	// Float32 upsampling to trigger DFTStage chunked path
	r32, err := NewResampler[float32](22050, 44100, QualityHigh)
	require.NoError(t, err)

	largeInput32 := make([]float32, 5000)
	for i := range largeInput32 {
		largeInput32[i] = 0.05
	}

	out32, err := r32.Process(largeInput32)
	require.NoError(t, err)
	assert.Len(t, out32, 9835)

	// DFT Decimation stage large input
	rDecim, err := NewResampler[float64](44100, 22050, QualityHigh)
	require.NoError(t, err)

	outDecim, err := rDecim.Process(largeInput64)
	require.NoError(t, err)
	// Aligned decimation prefills (751-1)/2 = 375 zeros, so the stage sees
	// 5375 samples: (5375 - 751 + 1) = 4625 filter positions, every second
	// one output: ceil(4625 / 2) = 2313 output samples
	assert.Len(t, outDecim, 2313)
}
