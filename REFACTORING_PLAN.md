# Refactoring Plan: `resampleWAVGeneric` Cognitive Complexity Reduction

> **Note:** This document describes the initial refactoring strategy and design. Code snippets represent the planned approach; actual implementation may differ based on discoveries during development. Refer to the source code for the final implementation.

## Problem Statement

**File**: [cmd/resample-wav/main.go:217](cmd/resample-wav/main.go#L217)
**Function**: `resampleWAVGeneric[F Float]`
**Current Complexity**: 64 (target: ≤50)
**Linter**: gocognit

## Current Structure Analysis

The function performs these logical operations (209 lines):

```text
resampleWAVGeneric[F Float]
├─ 1. Input file opening & validation (lines 218-244)
│  ├─ Open file
│  ├─ Create decoder
│  ├─ Validate WAV format
│  └─ Extract format info (rate, channels, bit depth)
│
├─ 2. Resampler creation (lines 246-254)
│  └─ Create one resampler per channel
│
├─ 3. Output file setup (lines 256-273)
│  ├─ Create output file
│  └─ Create fast WAV writer
│
├─ 4. Buffer initialization (lines 275-312)
│  ├─ Create stats struct
│  ├─ Calculate total samples
│  ├─ Allocate input buffer
│  ├─ Allocate per-channel buffers
│  ├─ Allocate resampled channels slice
│  ├─ Allocate output buffer
│  └─ Precompute normalization constants
│
├─ 5. Main processing loop (lines 314-390) ⚠️ HIGH COMPLEXITY
│  ├─ Read chunk
│  ├─ Update stats
│  ├─ Deinterleave
│  ├─ Resample channels (parallel OR sequential) ⚠️ NESTED CONDITIONALS
│  │  ├─ IF parallel && channels > 1:
│  │  │  ├─ Create WaitGroup
│  │  │  ├─ Launch goroutines for each channel
│  │  │  ├─ Error aggregation logic
│  │  │  └─ Wait for completion
│  │  └─ ELSE:
│  │     └─ Sequential loop over channels
│  ├─ Interleave output
│  ├─ Write samples
│  └─ Progress reporting (verbose && totalSamples > 0) ⚠️ NESTED CONDITIONALS
│
└─ 6. Flush remaining samples (lines 392-422) ⚠️ MODERATE COMPLEXITY
   ├─ Flush each channel
   ├─ Find max flush length
   ├─ Pad shorter channels
   ├─ Interleave flushed data
   └─ Write final samples
```

### Complexity Hotspots

1. **Main processing loop** (lines 314-390): ~35 complexity
   - Parallel/sequential branching
   - Error handling within goroutines
   - Progress calculation conditionals

2. **Flush logic** (lines 392-422): ~12 complexity
   - Channel padding logic
   - Multiple conditionals

3. **Setup sections**: ~17 combined complexity

## Proposed Refactoring Strategy

### Phase 1: Extract Input/Output Setup Helpers

**Goal**: Reduce setup complexity, improve testability

#### Helper 1: `wavInputInfo`

```go
// wavInputInfo holds validated input file information
type wavInputInfo struct {
    file      *os.File
    decoder   *wav.Decoder
    rate      int
    channels  int
    bitDepth  int
    totalSamples int64
}

// openWAVInput opens and validates a WAV file, returning format information
func openWAVInput(path string, verbose bool) (*wavInputInfo, error) {
    // Lines 218-244 logic
    // Returns: file, decoder, format info
    // Complexity: ~5
}

func (w *wavInputInfo) Close() error {
    return w.file.Close()
}
```

**Rationale**: Encapsulates all input validation. Testable with mock WAV files.

---

#### Helper 2: `createChannelResamplers`

```go
// createChannelResamplers creates one resampler per channel
func createChannelResamplers[F Float](
    numChannels int,
    inputRate, targetRate int,
    quality engine.Quality,
) ([]*engine.Resampler[F], error) {
    // Lines 246-254 logic
    // Complexity: ~3
}
```

**Rationale**: Single responsibility. Easy to test with various channel counts.

---

#### Helper 3: `wavOutputWriter`

```go
// wavOutputWriter wraps output file and fast writer
type wavOutputWriter struct {
    file   *os.File
    writer *fastWAVWriter
}

// createWAVOutput creates output file and writer
func createWAVOutput(
    path string,
    sampleRate, bitDepth, channels int,
) (*wavOutputWriter, error) {
    // Lines 256-273 logic
    // Complexity: ~4
}

func (w *wavOutputWriter) WriteSamples(samples []int) error {
    return w.writer.WriteSamples(samples)
}

func (w *wavOutputWriter) Close() error {
    if err := w.writer.Close(); err != nil {
        return err
    }
    return w.file.Close()
}
```

**Rationale**: Clean resource management. Testable output writer.

---

### Phase 2: Extract Processing State and Buffers

#### Helper 4: `resampleBuffers`

```go
// resampleBuffers holds all preallocated buffers for resampling
type resampleBuffers[F Float] struct {
    intBuffer         *audio.IntBuffer
    channelBufs       [][]F
    resampledChannels [][]F
    outputIntBuf      []int
    invMaxVal         float64
    maxVal            float64
}

// newResampleBuffers creates and preallocates all processing buffers
func newResampleBuffers[F Float](
    channels, bitDepth int,
    inputRate, targetRate int,
    format *audio.Format,
) *resampleBuffers[F] {
    // Lines 290-312 logic
    // Complexity: ~2
}
```

**Rationale**: Isolates buffer allocation. No complexity, pure setup.

---

### Phase 3: Extract Core Processing Logic

#### Helper 5: `resampleChannelData` (extracted from parallel/sequential branching)

```go
// resampleChannelData resamples channel buffers using provided resamplers.
// Handles both parallel and sequential modes based on config.
func resampleChannelData[F Float](
    resamplers []*engine.Resampler[F],
    channelBufs [][]F,
    numSamples int,
    parallel bool,
) ([][]F, error) {
    channels := len(resamplers)
    resampledChannels := make([][]F, channels)

    // Parallel processing for multichannel
    if parallel && channels > 1 {
        return resampleParallel(resamplers, channelBufs, numSamples, channels)
    }

    // Sequential processing
    return resampleSequential(resamplers, channelBufs, numSamples, channels)
}

// resampleParallel processes channels concurrently
func resampleParallel[F Float](
    resamplers []*engine.Resampler[F],
    channelBufs [][]F,
    numSamples, channels int,
) ([][]F, error) {
    // Lines 332-358 logic
    // Complexity: ~10
}

// resampleSequential processes channels one by one
func resampleSequential[F Float](
    resamplers []*engine.Resampler[F],
    channelBufs [][]F,
    numSamples, channels int,
) ([][]F, error) {
    // Lines 360-368 logic
    // Complexity: ~3
}
```

**Rationale**: Separates parallel/sequential logic. Each testable independently.

---

#### Helper 6: `progressTracker`

```go
// progressTracker handles progress reporting
type progressTracker struct {
    totalSamples int64
    lastProgress int
    verbose      bool
}

func newProgressTracker(totalSamples int64, verbose bool) *progressTracker {
    return &progressTracker{
        totalSamples: totalSamples,
        verbose:      verbose,
    }
}

// reportIfNeeded reports progress if threshold crossed
func (p *progressTracker) reportIfNeeded(currentSamples int64) {
    if !p.verbose || p.totalSamples == 0 {
        return
    }

    progress := int(float64(currentSamples) / float64(p.totalSamples) * percentScale)
    if progress >= p.lastProgress+progressInterval {
        log.Printf("Progress: %d%%", progress)
        p.lastProgress = progress
    }
}
```

**Rationale**: Removes conditionals from main loop. Testable progress logic.

---

#### Helper 7: `flushAndPadChannels`

```go
// flushAndPadChannels flushes all resamplers and pads channels to equal length
func flushAndPadChannels[F Float](
    resamplers []*engine.Resampler[F],
    bitDepth int,
) ([]int, int, error) {
    channels := len(resamplers)
    flushedData := make([][]F, channels)
    maxFlushLen := 0

    // Flush all channels
    for ch := range channels {
        flushed, err := resamplers[ch].Flush()
        if err != nil {
            return nil, 0, fmt.Errorf("failed to flush resampler channel %d: %w", ch, err)
        }
        flushedData[ch] = flushed
        if len(flushed) > maxFlushLen {
            maxFlushLen = len(flushed)
        }
    }

    if maxFlushLen == 0 {
        return nil, 0, nil
    }

    // Pad shorter channels
    for ch := range channels {
        if len(flushedData[ch]) < maxFlushLen {
            padded := make([]F, maxFlushLen)
            copy(padded, flushedData[ch])
            flushedData[ch] = padded
        }
    }

    outputData := interleaveGeneric(flushedData, bitDepth)
    return outputData, maxFlushLen, nil
}
```

**Rationale**: Isolates flush complexity. Testable with mock resamplers.

---

### Phase 4: Refactored Main Function

```go
func resampleWAVGeneric[F Float](
    inputPath, outputPath string,
    targetRate int,
    quality engine.Quality,
    verbose, parallel bool,
) (*resampleStats, error) {
    // 1. Open and validate input (~2 complexity)
    input, err := openWAVInput(inputPath, verbose)
    if err != nil {
        return nil, err
    }
    defer input.Close()

    if input.rate == targetRate {
        return nil, fmt.Errorf("input already at target rate %d Hz", targetRate)
    }

    // 2. Create resamplers (~2 complexity)
    resamplers, err := createChannelResamplers[F](
        input.channels, input.rate, targetRate, quality,
    )
    if err != nil {
        return nil, err
    }

    // 3. Create output writer (~2 complexity)
    output, err := createWAVOutput(
        outputPath, targetRate, input.bitDepth, input.channels,
    )
    if err != nil {
        return nil, err
    }
    defer output.Close()

    // 4. Initialize processing buffers (~1 complexity)
    buffers := newResampleBuffers[F](
        input.channels, input.bitDepth,
        input.rate, targetRate,
        input.decoder.Format(),
    )

    // 5. Initialize tracking (~1 complexity)
    stats := &resampleStats{
        inputRate:  input.rate,
        outputRate: targetRate,
        channels:   input.channels,
        bitDepth:   input.bitDepth,
    }
    progress := newProgressTracker(input.totalSamples, verbose)

    // 6. Main processing loop (~10 complexity)
    for {
        // Read chunk
        n, err := input.decoder.PCMBuffer(buffers.intBuffer)
        if err != nil && !errors.Is(err, io.EOF) {
            return nil, fmt.Errorf("failed to read audio data: %w", err)
        }
        if n == 0 {
            break
        }

        // Update tracking
        buffers.intBuffer.Data = buffers.intBuffer.Data[:n*input.channels]
        stats.inputSamples += int64(n)

        // Deinterleave
        deinterleaveInto(
            buffers.intBuffer.Data,
            buffers.channelBufs,
            input.channels, n,
            buffers.invMaxVal,
        )

        // Resample channels (handles parallel/sequential)
        resampledChannels, err := resampleChannelData(
            resamplers,
            buffers.channelBufs,
            n,
            parallel,
        )
        if err != nil {
            return nil, err
        }

        // Interleave and write
        outputLen := interleaveInto(
            resampledChannels,
            buffers.outputIntBuf,
            buffers.maxVal,
        )
        stats.outputSamples += int64(outputLen / input.channels)

        if err := output.WriteSamples(buffers.outputIntBuf[:outputLen]); err != nil {
            return nil, fmt.Errorf("failed to write audio data: %w", err)
        }

        // Progress reporting
        progress.reportIfNeeded(stats.inputSamples)

        // Reset buffer
        buffers.intBuffer.Data = buffers.intBuffer.Data[:cap(buffers.intBuffer.Data)]
    }

    // 7. Flush remaining samples (~3 complexity)
    flushedData, flushedSamples, err := flushAndPadChannels(resamplers, input.bitDepth)
    if err != nil {
        return nil, err
    }

    if flushedSamples > 0 {
        stats.outputSamples += int64(flushedSamples)
        if err := output.WriteSamples(flushedData); err != nil {
            return nil, fmt.Errorf("failed to write flushed data: %w", err)
        }
    }

    return stats, nil
}
```

**New Complexity**: ~22 (down from 64)

---

## Complexity Reduction Summary

| Component | Before | After | Location |
|-----------|--------|-------|----------|
| Main function | 64 | ~22 | Refactored |
| Input setup | - | ~5 | `openWAVInput` |
| Resampler creation | - | ~3 | `createChannelResamplers` |
| Output setup | - | ~4 | `createWAVOutput` |
| Buffer allocation | - | ~2 | `newResampleBuffers` |
| Parallel resampling | - | ~10 | `resampleParallel` |
| Sequential resampling | - | ~3 | `resampleSequential` |
| Progress tracking | - | ~2 | `progressTracker` |
| Flush & pad | - | ~8 | `flushAndPadChannels` |

**Total Complexity**: Distributed across 9 functions instead of 1
**Main Function**: 64 → 22 ✅

---

## Testing Strategy

### Phase 1: Baseline Tests
- [ ] Run existing tests and verify all pass
- [ ] Document baseline coverage (77.1%)

### Phase 2: Add Tests for New Helpers

#### `openWAVInput`
- [ ] Valid WAV file
- [ ] Invalid file (not WAV)
- [ ] File not found
- [ ] Corrupt header

#### `createChannelResamplers`
- [ ] Single channel (mono)
- [ ] Stereo
- [ ] Multi-channel (5.1, 7.1)
- [ ] Invalid rate (error case)

#### `createWAVOutput`
- [ ] Valid output path
- [ ] Invalid directory
- [ ] Permission denied

#### `resampleChannelData`
- [ ] Sequential mode, mono
- [ ] Sequential mode, stereo
- [ ] Parallel mode, mono (should fall back to sequential)
- [ ] Parallel mode, stereo
- [ ] Parallel mode, 8 channels
- [ ] Error from resampler

#### `progressTracker`
- [ ] Verbose mode, reports at intervals
- [ ] Non-verbose mode, no output
- [ ] Zero total samples, no panic

#### `flushAndPadChannels`
- [ ] All channels same length
- [ ] Channels different lengths (padding test)
- [ ] Empty flush (maxFlushLen = 0)
- [ ] Flush error propagation

### Phase 3: Integration Tests
- [ ] Full resampling workflow (existing tests)
- [ ] Verify output matches before refactoring
- [ ] Benchmark: ensure no performance regression

---

## Implementation Steps

### Step 1: Create Helper Functions File
Create `cmd/resample-wav/helpers.go` with all extracted helpers.

### Step 2: Implement Helpers One-by-One
For each helper:
1. Write tests first (TDD approach)
2. Implement helper
3. Verify tests pass
4. Run linter

### Step 3: Refactor Main Function
1. Replace sections with helper calls
2. Run tests after each replacement
3. Verify linter passes

### Step 4: Cleanup
1. Remove any dead code
2. Add godoc comments
3. Final linter check
4. Final test coverage check

---

## Success Criteria

- [ ] `resampleWAVGeneric` complexity ≤ 50 (target: ~22)
- [ ] All existing tests pass
- [ ] Coverage maintained or improved (≥77.1%)
- [ ] No performance regression (±5%)
- [ ] All helpers have unit tests
- [ ] Linter passes with 0 errors
- [ ] Code review approved

---

## Files to Modify

1. **cmd/resample-wav/main.go**
   - Refactor `resampleWAVGeneric`

2. **cmd/resample-wav/helpers.go** (NEW)
   - All extracted helper functions and types

3. **cmd/resample-wav/helpers_test.go** (NEW)
   - Unit tests for all helpers

---

## Estimated Impact

**Lines of Code**:
- Before: 1 function × 209 lines = 209 lines
- After: 10 functions × ~30 lines avg = ~300 lines (+45% for better structure)

**Maintainability**: ⬆️ Significantly improved
- Each function has single responsibility
- Testable in isolation
- Easier to debug
- Clear separation of concerns

**Performance**: → No change expected
- Same logic, just reorganized
- May benefit from better inlining opportunities

**Test Coverage**: ⬆️ Will increase
- Each helper independently testable
- More granular test cases possible
