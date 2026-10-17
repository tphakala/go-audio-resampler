// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"encoding/binary"
	"image"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

// writeInt16WAV writes mono 16-bit PCM samples (already integer valued) to a
// WAV file.
func writeInt16WAV(t *testing.T, path string, samples []float64, rate int) {
	t.Helper()
	const (
		headerLen    = 44
		fmtChunkLen  = 16
		bytesPerSamp = 2
	)
	dataLen := bytesPerSamp * len(samples)
	wav := make([]byte, 0, headerLen+dataLen)
	le := binary.LittleEndian
	wav = append(wav, "RIFF"...)
	wav = le.AppendUint32(wav, uint32(headerLen-8+dataLen))
	wav = append(wav, "WAVEfmt "...)
	wav = le.AppendUint32(wav, fmtChunkLen)
	wav = le.AppendUint16(wav, 1) // PCM
	wav = le.AppendUint16(wav, 1) // mono
	wav = le.AppendUint32(wav, uint32(rate))
	wav = le.AppendUint32(wav, uint32(rate*bytesPerSamp))
	wav = le.AppendUint16(wav, bytesPerSamp)
	wav = le.AppendUint16(wav, 8*bytesPerSamp)
	wav = append(wav, "data"...)
	wav = le.AppendUint32(wav, uint32(dataLen))
	for _, v := range samples {
		wav = le.AppendUint16(wav, uint16(int16(v)))
	}
	if err := os.WriteFile(path, wav, 0o600); err != nil {
		t.Fatal(err)
	}
}

// spectrogramClip is the analysis clip of a spectrogram consumer: tones and a
// chirp placed relative to the Nyquist, rounded to int16 values. The 0.8
// tone lies above the output Nyquist of both tested ratios (about 0.67), so
// without the anti-alias lowpass it folds into the passband.
func spectrogramClip(rate, secs int) []float64 {
	nyq := float64(rate) / 2
	out := make([]float64, rate*secs)
	for i := range out {
		tm := float64(i) / float64(rate)
		v := 0.25*math.Sin(2*math.Pi*0.12*nyq*tm) + 0.15*math.Sin(2*math.Pi*0.31*nyq*tm) +
			0.1*math.Sin(2*math.Pi*0.8*nyq*tm)
		v += 0.2 * math.Sin(2*math.Pi*(0.05*nyq+0.2*nyq*tm/float64(secs))*tm)
		out[i] = math.Round(v * math.MaxInt16)
	}
	return out
}

func decodePalettedPNG(t *testing.T, path string) *image.Paletted {
	t.Helper()
	f, err := os.Open(path) //nolint:gosec // path is under t.TempDir()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	im, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := im.(*image.Paletted)
	if !ok {
		t.Fatalf("%s: decoded %T, want *image.Paletted", path, im)
	}
	return p
}

// libraryF32Bytes resamples clip (int16-valued samples) from inRate to outRate
// with the float32 engine at QualityHigh, in 65536-sample ProcessInto chunks
// and one Flush as the consumer runs it, and returns the raw little-endian
// float32 output.
func libraryF32Bytes(t *testing.T, clip []float64, inRate, outRate int) []byte {
	t.Helper()
	r, err := NewEngineFloat32(float64(inRate), float64(outRate), QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]float32, r.EstimateOutput(antiAliasChunk))
	var raw []byte
	appendF32 := func(s []float32) {
		for _, v := range s {
			raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
		}
	}
	for off := 0; off < len(clip); off += antiAliasChunk {
		end := min(off+antiAliasChunk, len(clip))
		blk := make([]float32, end-off)
		for i := range blk {
			blk[i] = float32(clip[off+i] / (math.MaxInt16 + 1))
		}
		n, err := r.ProcessInto(blk, buf)
		if err != nil {
			t.Fatal(err)
		}
		appendF32(buf[:n])
	}
	tail, err := r.Flush()
	if err != nil {
		t.Fatal(err)
	}
	appendF32(tail)
	return raw
}

// runSox runs the sox binary with args and fails the test on error.
func runSox(t *testing.T, soxPath string, args ...string) {
	t.Helper()
	cmd := exec.CommandContext(t.Context(), soxPath, args...) //nolint:gosec // fixed binary, test-generated arguments
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("sox %v: %v\n%s", args, err, out)
	}
}

// compareRasters returns the fraction of equal pixels and the fraction more
// than bigSteps palette entries apart.
func compareRasters(t *testing.T, a, b *image.Paletted, bigSteps int) (equalFrac, bigFrac float64) {
	t.Helper()
	if a.Bounds() != b.Bounds() {
		t.Fatalf("raster sizes differ: %v vs %v", a.Bounds(), b.Bounds())
	}
	var equal, big, total int
	for y := a.Bounds().Min.Y; y < a.Bounds().Max.Y; y++ {
		for x := a.Bounds().Min.X; x < a.Bounds().Max.X; x++ {
			d := int(a.ColorIndexAt(x, y)) - int(b.ColorIndexAt(x, y))
			total++
			if d == 0 {
				equal++
			}
			if d > bigSteps || -d > bigSteps {
				big++
			}
		}
	}
	return float64(equal) / float64(total), float64(big) / float64(total)
}

// TestSoxSpectrogramParity_NonIntegerDownsampling renders a 120 dB sox
// spectrogram of a clip resampled by this library and of the same clip
// resampled by sox, and requires the rasters to agree. Before non-integer
// downsampling was low-passed at the target Nyquist, content above it folded
// into the spectrogram and about a quarter of the pixels differed.
func TestSoxSpectrogramParity_NonIntegerDownsampling(t *testing.T) {
	if testing.Short() {
		t.Skip("spectrogram parity renders with sox; skipped in -short mode")
	}
	soxPath, err := exec.LookPath("sox")
	if err != nil {
		t.Skip("sox not installed")
	}
	const (
		secs = 3
		// Measured raster-equal fractions with sox 14.4.2: amd64 0.9966
		// (384000 to 256000) and 0.9914 (48000 to 32000), arm64 0.9893 for the
		// latter, against 0.7285 and 0.7801 before the fix. The few percent of
		// unequal pixels are one palette step apart (rounding at a colour
		// boundary), which is why the second criterion counts pixels more
		// than bigSteps apart.
		minEqualFrac = 0.98
		maxBigFrac   = 0.001
		bigSteps     = 2
	)
	for _, rr := range [][2]int{{384000, 256000}, {48000, 32000}} {
		t.Run(strconv.Itoa(rr[0])+"_to_"+strconv.Itoa(rr[1]), func(t *testing.T) {
			dir := t.TempDir()
			clip := spectrogramClip(rr[0], secs)
			wav, lib := filepath.Join(dir, "ref.wav"), filepath.Join(dir, "lib.f32")
			writeInt16WAV(t, wav, clip, rr[0])
			if err := os.WriteFile(lib, libraryF32Bytes(t, clip, rr[0], rr[1]), 0o600); err != nil {
				t.Fatal(err)
			}

			spec := []string{"spectrogram", "-x", "514", "-y", "257", "-d", strconv.Itoa(secs), "-z", "120", "-h", "-r"}
			refPNG, libPNG := filepath.Join(dir, "ref.png"), filepath.Join(dir, "lib.png")
			out := strconv.Itoa(rr[1])
			runSox(t, soxPath, append([]string{wav, "-n", "rate", out}, append(spec, "-o", refPNG)...)...)
			runSox(t, soxPath, append([]string{"-t", "raw", "-e", "floating-point", "-b", "32", "-r", out, "-c", "1", lib, "-n"}, append(spec, "-o", libPNG)...)...)

			eqFrac, bigFrac := compareRasters(t, decodePalettedPNG(t, refPNG), decodePalettedPNG(t, libPNG), bigSteps)
			t.Logf("raster-equal %.4f, %.5f of pixels more than %d palette steps apart", eqFrac, bigFrac, bigSteps)
			if eqFrac < minEqualFrac {
				t.Errorf("raster-equal fraction %.4f, want >= %.2f", eqFrac, minEqualFrac)
			}
			if bigFrac > maxBigFrac {
				t.Errorf("%.5f of pixels more than %d palette steps apart, want <= %.4f", bigFrac, bigSteps, maxBigFrac)
			}
		})
	}
}
