// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvenienceConstructors(t *testing.T) {
	// NewCDtoDAT
	r, err := NewCDtoDAT(QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)
	assert.InDelta(t, float64(RateDAT)/float64(RateCD), r.GetRatio(), 1e-9)

	// NewDATtoCD
	r, err = NewDATtoCD(QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)
	assert.InDelta(t, float64(RateCD)/float64(RateDAT), r.GetRatio(), 1e-9)

	// NewCDtoHiRes
	r, err = NewCDtoHiRes(QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)
	assert.InDelta(t, float64(RateHiRes88)/float64(RateCD), r.GetRatio(), 1e-9)

	// NewHiRestoCD
	r, err = NewHiRestoCD(QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)
	assert.InDelta(t, float64(RateCD)/float64(RateHiRes88), r.GetRatio(), 1e-9)

	// NewSimple
	r, err = NewSimple(44100, 48000)
	require.NoError(t, err)
	assert.NotNil(t, r)
	assert.InDelta(t, 48000.0/44100.0, r.GetRatio(), 1e-9)

	// NewStereo
	r, err = NewStereo(44100, 48000, QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)

	// NewMultiChannel
	r, err = NewMultiChannel(44100, 48000, 6, QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, r)
}

func TestSimpleResamplerGetters(t *testing.T) {
	sr, err := NewEngine(44100, 48000, QualityHigh)
	require.NoError(t, err)
	assert.NotNil(t, sr)

	// GetRatio
	assert.InDelta(t, 48000.0/44100.0, sr.GetRatio(), 1e-9)

	// GetStatistics
	stats := sr.GetStatistics()
	assert.NotNil(t, stats)
}

func TestInterleaveDeinterleaveStereoFloat64(t *testing.T) {
	left := []float64{1.0, 2.0, 3.0}
	right := []float64{-1.0, -2.0, -3.0}

	interleaved := InterleaveToStereo(left, right)
	assert.Equal(t, []float64{1.0, -1.0, 2.0, -2.0, 3.0, -3.0}, interleaved)

	gotLeft, gotRight := DeinterleaveFromStereo(interleaved)
	assert.Equal(t, left, gotLeft)
	assert.Equal(t, right, gotRight)
}

func TestConfigValidation_EdgeCases(t *testing.T) {
	// Negative Input Rate
	_, err := New(&Config{
		InputRate:  -44100,
		OutputRate: 48000,
		Channels:   1,
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Channels < 1
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   0,
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Too many channels
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   257,
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Ratio too small
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 44100 / 300.0,
		Channels:   1,
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Custom quality - bad precision
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   1,
		Quality: QualitySpec{
			Preset:    QualityCustom,
			Precision: 7,
		},
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Custom quality - bad phase response
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   1,
		Quality: QualitySpec{
			Preset:        QualityCustom,
			Precision:     16,
			PhaseResponse: 101,
		},
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Custom quality - bad passband end
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   1,
		Quality: QualitySpec{
			Preset:        QualityCustom,
			Precision:     16,
			PhaseResponse: 50,
			PassbandEnd:   1.1,
		},
	})
	require.ErrorIs(t, err, ErrInvalidConfig)

	// Custom quality - bad stopband begin
	_, err = New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   1,
		Quality: QualitySpec{
			Preset:        QualityCustom,
			Precision:     16,
			PhaseResponse: 50,
			PassbandEnd:   0.9,
			StopbandBegin: 0.8,
		},
	})
	require.ErrorIs(t, err, ErrInvalidConfig)
}

func TestConstantRateResamplerResetGetInfo(t *testing.T) {
	r, err := New(&Config{
		InputRate:  44100,
		OutputRate: 48000,
		Channels:   2,
		Quality:    QualitySpec{Preset: QualityHigh},
	})
	require.NoError(t, err)

	// Call Reset
	r.Reset()

	// GetInfo
	info := GetInfo(r)
	assert.Equal(t, "multi-stage", info.Algorithm)
	assert.Positive(t, info.Latency)
	assert.Positive(t, info.MemoryUsage)

	// GetInfo fallback
	fallbackInfo := GetInfo(&dummyResampler{})
	assert.Equal(t, "unknown", fallbackInfo.Algorithm)
	assert.Equal(t, 42, fallbackInfo.Latency)
}

type dummyResampler struct{}

func (d *dummyResampler) Process(input []float64) ([]float64, error)              { return nil, nil }
func (d *dummyResampler) ProcessInto(input, output []float64) (int, error)        { return 0, nil }
func (d *dummyResampler) ProcessFloat32(input []float32) ([]float32, error)       { return nil, nil }
func (d *dummyResampler) ProcessFloat32Into(input, output []float32) (int, error) { return 0, nil }
func (d *dummyResampler) ProcessMulti(input [][]float64) ([][]float64, error)     { return nil, nil }
func (d *dummyResampler) Flush() ([]float64, error)                               { return nil, nil }
func (d *dummyResampler) Reset()                                                  {}
func (d *dummyResampler) GetRatio() float64                                       { return 1.0 }
func (d *dummyResampler) GetLatency() int                                         { return 42 }

func TestStubStage(t *testing.T) {
	s := &stubStage{
		ratio:        1.5,
		filterLength: 32,
		phases:       64,
		name:         "test-stub",
	}

	// Process
	input := []float64{1.0, 2.0, 3.0}
	output, err := s.Process(input)
	require.NoError(t, err)
	assert.Len(t, output, 4) // 3 * 1.5 = 4.5 -> 4

	// Flush
	flushed, err := s.Flush()
	require.NoError(t, err)
	assert.Empty(t, flushed)

	// Reset
	s.Reset()

	// Getters
	assert.InDelta(t, 1.5, s.GetRatio(), 1e-9)
	assert.Equal(t, 16, s.GetLatency()) // 32 / 2 = 16
	assert.Equal(t, 1, s.GetMinInput())
	assert.Equal(t, int64(32*8), s.GetMemoryUsage())
	assert.Equal(t, 32, s.GetFilterLength())
	assert.Equal(t, 64, s.GetPhases())
	assert.Empty(t, s.GetSIMDInfo())

	// StubStage filterLength 0 latency
	s2 := &stubStage{filterLength: 0}
	assert.Equal(t, 0, s2.GetLatency())
}
