//go:build arm64

#include "textflag.h"

// Fused polyphase cubic resampler kernels ported into this repo from
// github.com/tphakala/simd (removed there as application-specific). The
// phase-stepping block orchestration is specific to this resampler; the inner
// cubic dot is the same math as simd CubicInterpDot, inlined per output.
// Self-contained: the blocks reference no external symbols or data.

// func resampleCubic32NEON(out, hist []float32, a, b, c, d [][]float32,
//                                 at, step int64, numPhases, tapsPerPhase, fracBits int) int
//
// Fused polyphase cubic resampler (float32, NEON): runs the whole output block in
// one pass, reusing cubicInterpDotNEON's inner dot body (same V registers, same
// WORD-encoded FMLA/FADD/DUP) for each output so the per-output dot is
// bit-identical to the standalone kernel. Returns the number of outputs written.
//
// The outer stepping state lives in R19-R25 (permanent scratch), which the inner
// body (R0-R7, V0-V31) never touches. sPhase and sFrac are re-derived from step
// per output with a multiply, not a divide. x is built with SCVTFS + FMULS and
// broadcast with the same DUP encoding the standalone kernel uses. Reserved
// registers R28/R27/R18/R16/R17 are never touched.
//
// Persistent registers across the whole loop:
//   R19 = div
//   R20 = phase
//   R21 = frac
//   R22 = k (output index)
//   R23 = sDiv
//   R24 = numPhases
//   R25 = fracBits
//   F30 = fracScale (2^-fracBits, scalar)
//   V31 = x broadcast (F31 lane 0 = x for the scalar tail)
//
// Frame layout (6 slices + 2 int64 + 3 int + 1 return):
//   out:  base+0,  len+8
//   hist: base+24, len+32
//   a:    base+48
//   b:    base+72
//   c:    base+96
//   d:    base+120
//   at:          +144
//   step:        +152
//   numPhases:   +160
//   tapsPerPhase:+168
//   fracBits:    +176
//   ret:         +184
TEXT ·resampleCubic32NEON(SB), NOSPLIT, $0-192
    MOVD fracBits+176(FP), R25     // R25 = fracBits
    MOVD numPhases+160(FP), R24    // R24 = numPhases

    // sDiv = (step >> fracBits) / numPhases  (one division, hoisted)
    MOVD step+152(FP), R0
    LSR  R25, R0, R0               // R0 = sFull = step >> fracBits
    UDIV R24, R0, R23              // R23 = sDiv = sFull / numPhases

    // Seed div, phase, frac from at.
    MOVD at+144(FP), R21           // R21 = at (masked to frac below)
    LSR  R25, R21, R0              // R0 = full = at >> fracBits
    UDIV R24, R0, R19              // R19 = div = full / numPhases
    MUL  R24, R19, R1              // R1 = div*numPhases
    SUB  R1, R0, R20               // R20 = phase = full - div*numPhases
    MOVD $1, R1
    LSL  R25, R1, R1               // R1 = 1<<fracBits
    SUB  $1, R1, R1                // R1 = fracMask
    AND  R1, R21, R21              // R21 = frac = at & fracMask

    // fracScale = 2^-fracBits as float32: bits = (127 - fracBits) << 23.
    MOVD $127, R0
    SUB  R25, R0, R0
    LSL  $23, R0, R0
    FMOVS R0, F30                  // F30 = fracScale

    MOVD $0, R22                   // k = 0

prcubic_neon_loop:
    MOVD out_len+8(FP), R7
    CMP  R7, R22
    BGE  prcubic_neon_done         // k >= len(out)
    MOVD tapsPerPhase+168(FP), R6  // R6 = taps (inner dot length)
    ADD  R6, R19, R8               // R8 = div + taps
    MOVD hist_len+32(FP), R7
    CMP  R7, R8
    BGT  prcubic_neon_done         // div + taps > len(hist)

    // x = float32(frac) * fracScale, broadcast to V31.
    SCVTFS R21, F31                // F31 = float32(frac)
    FMULS  F30, F31, F31           // F31 = x (lane 0)
    WORD $0x4E0407FF               // DUP V31.4S, V31.S[0]

    // Row pointers: byte offset = phase*24.
    MOVD $24, R7
    MUL  R7, R20, R8               // R8 = phase*24
    MOVD a_base+48(FP), R9
    MOVD (R9)(R8), R1              // R1 = &a[phase][0]
    MOVD b_base+72(FP), R9
    MOVD (R9)(R8), R2             // R2 = &b[phase][0]
    MOVD c_base+96(FP), R9
    MOVD (R9)(R8), R3             // R3 = &c[phase][0]
    MOVD d_base+120(FP), R9
    MOVD (R9)(R8), R4             // R4 = &d[phase][0]
    MOVD hist_base+24(FP), R9
    ADD  R19<<2, R9, R0           // R0 = &hist[div]

    // ---- inner dot body, V registers verbatim from cubicInterpDotNEON ----
    VEOR V0.B16, V0.B16, V0.B16   // acc0 = 0
    VEOR V1.B16, V1.B16, V1.B16   // acc1 = 0
    LSR  $3, R6, R5               // R5 = taps / 8
    CBZ  R5, prcubic_neon_loop4_check

prcubic_neon_loop8:
    VLD1.P 16(R4), [V2.S4]
    VLD1.P 16(R3), [V3.S4]
    VLD1.P 16(R2), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    VLD1.P 16(R0), [V6.S4]
    VLD1.P 16(R4), [V10.S4]
    VLD1.P 16(R3), [V11.S4]
    VLD1.P 16(R2), [V12.S4]
    VLD1.P 16(R1), [V13.S4]
    VLD1.P 16(R0), [V14.S4]
    WORD $0x4E3FCC43              // FMLA V3.4S, V2.4S, V31.4S
    WORD $0x4E3FCC64              // FMLA V4.4S, V3.4S, V31.4S
    WORD $0x4E3FCC85              // FMLA V5.4S, V4.4S, V31.4S
    WORD $0x4E25CCC0              // FMLA V0.4S, V6.4S, V5.4S
    WORD $0x4E3FCD4B              // FMLA V11.4S, V10.4S, V31.4S
    WORD $0x4E3FCD6C              // FMLA V12.4S, V11.4S, V31.4S
    WORD $0x4E3FCD8D              // FMLA V13.4S, V12.4S, V31.4S
    WORD $0x4E2DCDC1              // FMLA V1.4S, V14.4S, V13.4S
    SUB  $1, R5
    CBNZ R5, prcubic_neon_loop8
    WORD $0x4E21D400             // FADD V0.4S, V0.4S, V1.4S

prcubic_neon_loop4_check:
    AND  $7, R6, R5
    LSR  $2, R5, R7              // R7 = remainder / 4
    CBZ  R7, prcubic_neon_remainder
    VLD1.P 16(R4), [V2.S4]
    VLD1.P 16(R3), [V3.S4]
    VLD1.P 16(R2), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    VLD1.P 16(R0), [V6.S4]
    WORD $0x4E3FCC43              // FMLA V3.4S, V2.4S, V31.4S
    WORD $0x4E3FCC64              // FMLA V4.4S, V3.4S, V31.4S
    WORD $0x4E3FCC85              // FMLA V5.4S, V4.4S, V31.4S
    WORD $0x4E25CCC0              // FMLA V0.4S, V6.4S, V5.4S

prcubic_neon_remainder:
    WORD $0x6E20D400             // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800             // FADDP S0, V0.2S
    AND  $3, R5, R7
    CBZ  R7, prcubic_neon_store

prcubic_neon_scalar:
    FMOVS (R4), F2
    FMOVS (R3), F3
    FMOVS (R2), F4
    FMOVS (R1), F5
    FMOVS (R0), F6
    FMADDS F31, F3, F2, F3       // F3 = d*x + c
    FMADDS F31, F4, F3, F4       // F4 = (d*x+c)*x + b
    FMADDS F31, F5, F4, F5       // F5 = coef
    FMADDS F5, F0, F6, F0        // F0 = hist*coef + acc
    ADD  $4, R0
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R3
    ADD  $4, R4
    SUB  $1, R7
    CBNZ R7, prcubic_neon_scalar

prcubic_neon_store:
    MOVD out_base+0(FP), R8
    ADD  R22<<2, R8, R8
    FMOVS F0, (R8)                // out[k] = result

    // ---- advance the state machine ----
    // sFrac = step & fracMask;  frac += sFrac
    MOVD step+152(FP), R0
    MOVD $1, R1
    LSL  R25, R1, R1              // R1 = 1<<fracBits
    SUB  $1, R1, R2              // R2 = fracMask
    AND  R2, R0, R0             // R0 = step & fracMask = sFrac
    ADD  R0, R21, R21            // frac += sFrac
    CMP  R2, R21                 // frac - fracMask
    BLE  prcubic_neon_nocarry
    SUB  R1, R21, R21            // frac -= (1<<fracBits)
    ADD  $1, R20, R20            // phase++
prcubic_neon_nocarry:
    // sPhase = (step>>fracBits) - sDiv*numPhases;  phase += sPhase;  div += sDiv
    MOVD step+152(FP), R0
    LSR  R25, R0, R0             // R0 = sFull
    MUL  R24, R23, R1            // R1 = sDiv*numPhases
    SUB  R1, R0, R0             // R0 = sPhase
    ADD  R0, R20, R20           // phase += sPhase
    ADD  R23, R19, R19          // div += sDiv
    CMP  R24, R20                // phase - numPhases
    BLT  prcubic_neon_nonorm
    SUB  R24, R20, R20
    ADD  $1, R19, R19
prcubic_neon_nonorm:
    ADD  $1, R22, R22            // k++
    B    prcubic_neon_loop

prcubic_neon_done:
    MOVD R22, ret+184(FP)
    RET

// func resampleCubic64NEON(out, hist []float64, a, b, c, d [][]float64,
//                                 at, step int64, numPhases, tapsPerPhase, fracBits int) int
//
// Fused polyphase cubic resampler (float64, NEON): runs the whole output block in
// one pass, reusing cubicInterpDotNEON's inner dot body (same V registers, same
// WORD-encoded FMLA/FADD/DUP) for each output so the per-output dot is
// bit-identical to the standalone kernel. Returns the number of outputs written.
//
// The outer stepping state lives in R19-R25 (permanent scratch), which the inner
// body (R0-R7, V0-V31) never touches. sPhase and sFrac are re-derived from step
// per output with a multiply, not a divide. Reserved registers
// R28/R27/R18/R16/R17 are never touched.
//
// Persistent registers across the whole loop:
//   R19 = div,  R20 = phase,  R21 = frac,  R22 = k,  R23 = sDiv,
//   R24 = numPhases,  R25 = fracBits,  F30 = fracScale,  V31 = x broadcast.
//
// Frame layout (6 slices + 2 int64 + 3 int + 1 return): out@0, hist@24, a@48,
// b@72, c@96, d@120, at@144, step@152, numPhases@160, tapsPerPhase@168,
// fracBits@176, ret@184.
TEXT ·resampleCubic64NEON(SB), NOSPLIT, $0-192
    MOVD fracBits+176(FP), R25     // R25 = fracBits
    MOVD numPhases+160(FP), R24    // R24 = numPhases

    // sDiv = (step >> fracBits) / numPhases  (one division, hoisted)
    MOVD step+152(FP), R0
    LSR  R25, R0, R0
    UDIV R24, R0, R23              // R23 = sDiv

    // Seed div, phase, frac from at.
    MOVD at+144(FP), R21
    LSR  R25, R21, R0              // R0 = full = at >> fracBits
    UDIV R24, R0, R19              // R19 = div
    MUL  R24, R19, R1
    SUB  R1, R0, R20               // R20 = phase
    MOVD $1, R1
    LSL  R25, R1, R1
    SUB  $1, R1, R1               // R1 = fracMask
    AND  R1, R21, R21             // R21 = frac

    // fracScale = 2^-fracBits as float64: bits = (1023 - fracBits) << 52.
    MOVD $1023, R0
    SUB  R25, R0, R0
    LSL  $52, R0, R0
    FMOVD R0, F30                  // F30 = fracScale

    MOVD $0, R22                   // k = 0

prcubic_neon_loop:
    MOVD out_len+8(FP), R7
    CMP  R7, R22
    BGE  prcubic_neon_done
    MOVD tapsPerPhase+168(FP), R6  // R6 = taps
    ADD  R6, R19, R8
    MOVD hist_len+32(FP), R7
    CMP  R7, R8
    BGT  prcubic_neon_done

    // x = float64(frac) * fracScale, broadcast to V31.
    SCVTFD R21, F31                // F31 = float64(frac)
    FMULD  F30, F31, F31           // F31 = x (lane 0)
    WORD $0x4E0807FF               // DUP V31.2D, V31.D[0]

    // Row pointers: byte offset = phase*24.
    MOVD $24, R7
    MUL  R7, R20, R8
    MOVD a_base+48(FP), R9
    MOVD (R9)(R8), R1             // &a[phase][0]
    MOVD b_base+72(FP), R9
    MOVD (R9)(R8), R2             // &b[phase][0]
    MOVD c_base+96(FP), R9
    MOVD (R9)(R8), R3             // &c[phase][0]
    MOVD d_base+120(FP), R9
    MOVD (R9)(R8), R4             // &d[phase][0]
    MOVD hist_base+24(FP), R9
    ADD  R19<<3, R9, R0           // &hist[div]  (float64 = 8 bytes)

    // ---- inner dot body, V registers verbatim from cubicInterpDotNEON ----
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    LSR  $2, R6, R5              // R5 = taps / 4
    CBZ  R5, prcubic_neon_loop2_check

prcubic_neon_loop4:
    VLD1.P 16(R4), [V2.D2]
    VLD1.P 16(R3), [V3.D2]
    VLD1.P 16(R2), [V4.D2]
    VLD1.P 16(R1), [V5.D2]
    VLD1.P 16(R0), [V6.D2]
    VLD1.P 16(R4), [V10.D2]
    VLD1.P 16(R3), [V11.D2]
    VLD1.P 16(R2), [V12.D2]
    VLD1.P 16(R1), [V13.D2]
    VLD1.P 16(R0), [V14.D2]
    WORD $0x4E7FCC43             // FMLA V3.2D, V2.2D, V31.2D
    WORD $0x4E7FCC64             // FMLA V4.2D, V3.2D, V31.2D
    WORD $0x4E7FCC85             // FMLA V5.2D, V4.2D, V31.2D
    WORD $0x4E65CCC0             // FMLA V0.2D, V6.2D, V5.2D
    WORD $0x4E7FCD4B             // FMLA V11.2D, V10.2D, V31.2D
    WORD $0x4E7FCD6C             // FMLA V12.2D, V11.2D, V31.2D
    WORD $0x4E7FCD8D             // FMLA V13.2D, V12.2D, V31.2D
    WORD $0x4E6DCDC1             // FMLA V1.2D, V14.2D, V13.2D
    SUB  $1, R5
    CBNZ R5, prcubic_neon_loop4
    WORD $0x4E61D400            // FADD V0.2D, V0.2D, V1.2D

prcubic_neon_loop2_check:
    AND  $3, R6, R5
    LSR  $1, R5, R7             // R7 = remainder / 2
    CBZ  R7, prcubic_neon_remainder1
    VLD1.P 16(R4), [V2.D2]
    VLD1.P 16(R3), [V3.D2]
    VLD1.P 16(R2), [V4.D2]
    VLD1.P 16(R1), [V5.D2]
    VLD1.P 16(R0), [V6.D2]
    WORD $0x4E7FCC43             // FMLA V3.2D, V2.2D, V31.2D
    WORD $0x4E7FCC64             // FMLA V4.2D, V3.2D, V31.2D
    WORD $0x4E7FCC85             // FMLA V5.2D, V4.2D, V31.2D
    WORD $0x4E65CCC0             // FMLA V0.2D, V6.2D, V5.2D

prcubic_neon_remainder1:
    WORD $0x7E70D800            // FADDP D0, V0.2D
    AND  $1, R5, R7
    CBZ  R7, prcubic_neon_store

    FMOVD (R4), F2
    FMOVD (R3), F3
    FMOVD (R2), F4
    FMOVD (R1), F5
    FMOVD (R0), F6
    FMADDD F31, F3, F2, F3       // F3 = d*x + c
    FMADDD F31, F4, F3, F4       // F4 = (d*x+c)*x + b
    FMADDD F31, F5, F4, F5       // F5 = coef
    FMADDD F5, F0, F6, F0        // F0 = hist*coef + acc

prcubic_neon_store:
    MOVD out_base+0(FP), R8
    ADD  R22<<3, R8, R8
    FMOVD F0, (R8)               // out[k] = result

    // ---- advance the state machine ----
    MOVD step+152(FP), R0
    MOVD $1, R1
    LSL  R25, R1, R1
    SUB  $1, R1, R2             // R2 = fracMask
    AND  R2, R0, R0            // R0 = sFrac
    ADD  R0, R21, R21           // frac += sFrac
    CMP  R2, R21
    BLE  prcubic_neon_nocarry
    SUB  R1, R21, R21           // frac -= (1<<fracBits)
    ADD  $1, R20, R20           // phase++
prcubic_neon_nocarry:
    MOVD step+152(FP), R0
    LSR  R25, R0, R0            // sFull
    MUL  R24, R23, R1           // sDiv*numPhases
    SUB  R1, R0, R0
    ADD  R0, R20, R20           // phase += sPhase
    ADD  R23, R19, R19          // div += sDiv
    CMP  R24, R20
    BLT  prcubic_neon_nonorm
    SUB  R24, R20, R20
    ADD  $1, R19, R19
prcubic_neon_nonorm:
    ADD  $1, R22, R22
    B    prcubic_neon_loop

prcubic_neon_done:
    MOVD R22, ret+184(FP)
    RET
