// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"bufio"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// goldenFile holds output lengths and output checksums captured from the
// code before output was time-aligned (golden_master.txt, one record per
// line). Lines:
//
//	len <path> <in> <out> <preset> <n> <count>
//	sum <path> <in> <out> <preset> <n> <fnv64a of the output bits>
//	sum-arm64 ...                                  (same, captured on arm64)
//
// The cubic stage and the float64 filter design are plain Go, and the
// compiler fuses multiply-adds on arm64, so output bits differ between amd64
// and arm64; the checksum records are therefore kept per architecture and
// skipped on others.
//
// The length records pin that alignment did not change how many samples a
// stream produces. The checksum records pin that QualityQuick and unity
// conversions, which were already aligned, are untouched, and that engine-path
// upsampling and integer decimation keep their bits when the non-integer
// downsampling pre-stage is redesigned (goldenSumFIRRatios).
const goldenFile = "testdata/golden_master.txt"

var (
	goldenLenRatios = [][2]float64{
		{48000, 24000}, {44100, 48000}, {48000, 44100}, {48000, 96000},
		{22050, 48000}, {8000, 48000}, {384000, 256000}, {48000, 32000},
		{16000, 48000}, {48000, 16000}, {44100, 64000}, {32000, 44100},
		{48000, 48000}, {11025, 44100}, {96000, 22050},
	}
	goldenLenPresets = []QualityPreset{QualityQuick, QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}
	goldenLenSizes   = []int{1, 2, 5, 17, 100, 1001, 4801}
	goldenChunkings  = []int{0, 1, 7, 333} // 0 means one call with the whole input
	goldenPaths      = []string{"eng64", "eng32", "new"}

	goldenSumQuickRatios = [][2]float64{
		{44100, 48000}, {48000, 44100}, {48000, 16000}, {8000, 48000}, {384000, 256000},
	}
	goldenSumUnityRates = []float64{8000, 44100, 48000}
	// goldenSumFIRRatios are the FIR-stage conversions that must stay
	// bit-identical when the non-integer downsampling pre-stage is redesigned:
	// engine-path upsampling and integer decimation. 48000 to 16000 is on the
	// list for the engine paths only: New factors a 2:1 half-band stage first
	// and hands the non-integer remainder to the engine, so its bits change.
	goldenSumFIRRatios = [][2]float64{
		{44100, 48000}, {8000, 48000}, {22050, 48000}, {48000, 96000},
		{32000, 44100}, {44100, 64000},
		{48000, 24000}, {96000, 48000}, {48000, 16000},
	}
	goldenSumFIRPresets = []QualityPreset{QualityLow, QualityMedium, QualityHigh, QualityVeryHigh}
	goldenSumSize       = 4801
)

// goldenSignal is a deterministic, exactly representable test signal (small
// rational values) so the input bits do not depend on the architecture's
// math library or FMA fusing.
func goldenSignal(n int) []float64 {
	in := make([]float64, n)
	for i := range in {
		in[i] = float64((i*37+11)%101-50) / 64
	}
	return in
}

// goldenRun streams in through the named path in chunks of the given size
// (0 = whole input) and flushes once. ok is false if the path cannot be
// constructed for this ratio.
func goldenRun(path string, inRate, outRate float64, q QualityPreset, in []float64, chunk int) (out []float64, ok bool, err error) {
	if chunk <= 0 {
		chunk = len(in)
	}
	switch path {
	case "eng64":
		r, err := NewEngine(inRate, outRate, q)
		if err != nil {
			return nil, false, nil //nolint:nilerr // an unsupported ratio simply has no golden row
		}
		for i := 0; i < len(in); i += chunk {
			o, err := r.Process(in[i:min(i+chunk, len(in))])
			if err != nil {
				return nil, true, err
			}
			out = append(out, o...)
		}
		t, err := r.Flush()
		return append(out, t...), true, err
	case "eng32":
		r, err := NewEngineFloat32(inRate, outRate, q)
		if err != nil {
			return nil, false, nil //nolint:nilerr // an unsupported ratio simply has no golden row
		}
		in32 := make([]float32, len(in))
		for i, v := range in {
			in32[i] = float32(v)
		}
		for i := 0; i < len(in32); i += chunk {
			o, err := r.Process(in32[i:min(i+chunk, len(in32))])
			if err != nil {
				return nil, true, err
			}
			for _, v := range o {
				out = append(out, float64(v))
			}
		}
		t, err := r.Flush()
		for _, v := range t {
			out = append(out, float64(v))
		}
		return out, true, err
	case "new":
		cfg := Config{InputRate: inRate, OutputRate: outRate, Channels: 1, Quality: QualitySpec{Preset: q}}
		r, err := New(&cfg)
		if err != nil {
			return nil, false, nil //nolint:nilerr // an unsupported ratio simply has no golden row
		}
		for i := 0; i < len(in); i += chunk {
			o, err := r.Process(in[i:min(i+chunk, len(in))])
			if err != nil {
				return nil, true, err
			}
			out = append(out, o...)
		}
		t, err := r.Flush()
		return append(out, t...), true, err
	}
	return nil, false, fmt.Errorf("unknown path %q", path)
}

func goldenChecksum(out []float64) uint64 {
	h := fnv.New64a()
	var b [8]byte
	for _, v := range out {
		u := math.Float64bits(v)
		for k := range b {
			b[k] = byte(u >> (8 * k))
		}
		_, _ = h.Write(b[:])
	}
	return h.Sum64()
}

func goldenKey(kind, path string, inRate, outRate float64, q QualityPreset, n int) string {
	return fmt.Sprintf("%s %s %g %g %d %d", kind, path, inRate, outRate, q, n)
}

// loadGolden reads the golden records into a map from key to value.
func loadGolden(t *testing.T) map[string]string {
	t.Helper()
	f, err := os.Open(goldenFile)
	if err != nil {
		t.Fatalf("open %s: %v", goldenFile, err)
	}
	defer func() { _ = f.Close() }()
	m := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 7 {
			continue
		}
		m[strings.Join(fields[:6], " ")] = fields[6]
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return m
}

// TestOutputLength_MatchesMaster pins the exact output count of a stream
// (Process calls plus one Flush) to the count produced before output was
// time-aligned, for every chunking. Callers size buffers and compute
// durations from it, so alignment must not change it.
func TestOutputLength_MatchesMaster(t *testing.T) {
	golden := loadGolden(t)
	chunkings := goldenChunkings
	if testing.Short() {
		// Keep the race-detector run quick: whole input and one small chunk.
		chunkings = []int{0, 7}
	}
	checked := 0
	for _, path := range goldenPaths {
		for _, rr := range goldenLenRatios {
			for _, q := range goldenLenPresets {
				for _, n := range goldenLenSizes {
					key := goldenKey("len", path, rr[0], rr[1], q, n)
					want, has := golden[key]
					in := goldenSignal(n)
					for _, chunk := range chunkings {
						out, ok, err := goldenRun(path, rr[0], rr[1], q, in, chunk)
						if err != nil {
							t.Fatalf("%s chunk=%d: %v", key, chunk, err)
						}
						if !ok {
							if has {
								t.Errorf("%s: golden row exists but constructor now fails", key)
							}
							break
						}
						if !has {
							t.Fatalf("%s: no golden row", key)
						}
						if got := fmt.Sprint(len(out)); got != want {
							t.Errorf("%s chunk=%d: output length %s, golden %s", key, chunk, got, want)
						}
						checked++
					}
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no golden rows checked")
	}
}

// goldenSumRow is one (ratio, preset) pair whose output bits are pinned.
// skipNew excludes the New(&Config) path for rows whose output on that path
// is expected to change legitimately.
type goldenSumRow struct {
	in, out float64
	q       QualityPreset
	skipNew bool
}

// goldenSumRows lists the (ratio, preset) pairs whose output bits are pinned.
func goldenSumRows() (rows []goldenSumRow) {
	for _, rr := range goldenSumQuickRatios {
		rows = append(rows, goldenSumRow{in: rr[0], out: rr[1], q: QualityQuick})
	}
	for _, rate := range goldenSumUnityRates {
		for _, q := range goldenLenPresets {
			rows = append(rows, goldenSumRow{in: rate, out: rate, q: q})
		}
	}
	for _, rr := range goldenSumFIRRatios {
		for _, q := range goldenSumFIRPresets {
			rows = append(rows, goldenSumRow{in: rr[0], out: rr[1], q: q, skipNew: rr[1] == 16000})
		}
	}
	return rows
}

// goamd64Fuses reports whether this binary was built with GOAMD64=v3 or v4,
// the levels at which the Go compiler emits fused multiply-add on amd64.
func goamd64Fuses() bool {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return false
	}
	for _, s := range info.Settings {
		if s.Key == "GOAMD64" {
			return s.Value == "v3" || s.Value == "v4"
		}
	}
	return false
}

// TestQuickAndUnity_BitIdenticalToMaster pins the output bits of the
// QualityQuick (cubic) preset and of unity conversions at every preset.
// Both were already time-aligned, so aligning the FIR stages must not touch
// them.
func TestQuickAndUnity_BitIdenticalToMaster(t *testing.T) {
	kind := "sum"
	switch runtime.GOARCH {
	case "amd64":
	case "arm64":
		kind = "sum-arm64"
	default:
		t.Skipf("no checksum golden for %s", runtime.GOARCH)
	}
	// From GOAMD64=v3 the compiler fuses multiply-adds on amd64 as it does on
	// arm64, so the cubic stage and the filter design produce bits that differ
	// from the amd64 golden (measured: the FIR rows fail under GOAMD64=v3).
	// Unity rows run neither and stay checked.
	fused := runtime.GOARCH == "amd64" && goamd64Fuses()
	golden := loadGolden(t)
	in := goldenSignal(goldenSumSize)
	for _, row := range goldenSumRows() {
		if fused && row.in != row.out {
			continue
		}
		for _, path := range goldenPaths {
			if row.skipNew && path == "new" {
				continue
			}
			key := goldenKey(kind, path, row.in, row.out, row.q, goldenSumSize)
			want, has := golden[key]
			out, ok, err := goldenRun(path, row.in, row.out, row.q, in, 333)
			if err != nil {
				t.Fatalf("%s: %v", key, err)
			}
			if !ok {
				if has {
					t.Errorf("%s: golden row exists but constructor now fails", key)
				}
				continue
			}
			if !has {
				t.Fatalf("%s: no golden row", key)
			}
			if got := fmt.Sprintf("%016x", goldenChecksum(out)); got != want {
				t.Errorf("%s: output checksum %s, golden %s", key, got, want)
			}
		}
	}
}
