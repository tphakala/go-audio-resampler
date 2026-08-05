//go:build amd64

#include "textflag.h"

// Fused polyphase cubic resampler kernels ported into this repo from
// github.com/tphakala/simd (removed there as application-specific). The
// phase-stepping block orchestration is specific to this resampler; the inner
// cubic dot is the same math as simd CubicInterpDot, inlined per output.
// Self-contained: the blocks reference no external symbols or data.

// func resampleCubic32AVX(out, hist []float32, a, b, c, d [][]float32,
//                                at, step int64, numPhases, tapsPerPhase, fracBits int) int
//
// Fused polyphase cubic resampler: runs the whole output block in one pass,
// reusing cubicInterpDotAVX's inner dot body verbatim for each output. Returns
// the number of outputs written.
//
// The outer stepping state machine keeps div, phase, frac, k and the delta sDiv
// in registers across the inner dot (which only clobbers SI/DI/R8/R9/R10/CX/AX
// and Y0-Y7); loop-invariant scalars are reloaded from FP. sPhase and sFrac are
// re-derived from step per output with a multiply, not a divide, so no division
// runs in the hot loop. x is splat AVX1-clean (VSHUFPS + VINSERTF128, no
// register-source VBROADCASTSS). Reserved registers R14/R15/BP are never touched.
//
// Persistent registers across the whole loop:
//   BX  = div
//   DX  = phase
//   R11 = frac
//   R12 = k (output index)
//   R13 = sDiv
//   X8  = fracScale (2^-fracBits, scalar)
//
// Frame layout (6 slices + 2 int64 + 3 int + 1 return):
//   out:  base+0,  len+8,  cap+16
//   hist: base+24, len+32, cap+40
//   a:    base+48, len+56, cap+64
//   b:    base+72, len+80, cap+88
//   c:    base+96, len+104, cap+112
//   d:    base+120,len+128, cap+136
//   at:          +144 (int64)
//   step:        +152 (int64)
//   numPhases:   +160 (int)
//   tapsPerPhase:+168 (int)
//   fracBits:    +176 (int)
//   ret:         +184 (int)
TEXT ·resampleCubic32AVX(SB), NOSPLIT, $0-192
    MOVQ fracBits+176(FP), CX      // CX = fracBits (shift count)
    MOVQ numPhases+160(FP), R8     // R8 = numPhases

    // sDiv = (step >> fracBits) / numPhases  (one division, hoisted out of loop)
    MOVQ step+152(FP), AX
    SARQ CX, AX                    // AX = sFull = step >> fracBits
    XORQ DX, DX
    DIVQ R8                        // AX = sDiv, DX = sPhase (discarded; re-derived)
    MOVQ AX, R13                   // R13 = sDiv

    // Seed div, phase, frac from at.
    MOVQ at+144(FP), R11           // R11 = at (reused as frac after masking)
    MOVQ R11, AX
    SARQ CX, AX                    // AX = full = at >> fracBits
    XORQ DX, DX
    DIVQ R8                        // AX = div, DX = phase
    MOVQ AX, BX                    // BX = div
                                   // DX = phase (kept)
    MOVQ $1, SI
    SHLQ CX, SI                    // SI = 1 << fracBits
    DECQ SI                        // SI = fracMask
    ANDQ SI, R11                   // R11 = frac = at & fracMask

    // fracScale = 2^-fracBits as float32: bits = (127 - fracBits) << 23.
    MOVQ $127, AX
    SUBQ CX, AX
    SHLQ $23, AX
    VMOVD AX, X8                   // X8 = fracScale (scalar lane 0)

    XORQ R12, R12                  // k = 0

prcubic_loop:
    // k < len(out) ?
    MOVQ out_len+8(FP), AX
    CMPQ R12, AX
    JGE  prcubic_done
    // div + tapsPerPhase <= len(hist) ?  (else stop)
    MOVQ tapsPerPhase+168(FP), CX  // CX = taps (also the inner dot length)
    MOVQ BX, AX
    ADDQ CX, AX                    // AX = div + taps
    MOVQ hist_len+32(FP), SI
    CMPQ AX, SI
    JGT  prcubic_done

    // x = float32(frac) * fracScale, broadcast to Y7 (AVX1-clean).
    VCVTSI2SSQ R11, X7, X7         // X7 = float32(frac)
    VMULSS   X8, X7, X7            // X7 = x (lane 0)
    VSHUFPS  $0, X7, X7, X7        // X7 = [x,x,x,x]
    VINSERTF128 $1, X7, Y7, Y7     // Y7 = [x x x x x x x x]

    // Row pointers for this phase: byte offset = phase*24 (Go slice header size).
    LEAQ (DX)(DX*2), AX            // AX = 3*phase
    SHLQ $3, AX                    // AX = 24*phase
    MOVQ a_base+48(FP), SI
    MOVQ (SI)(AX*1), DI            // DI = &a[phase][0]
    MOVQ b_base+72(FP), SI
    MOVQ (SI)(AX*1), R8            // R8 = &b[phase][0]
    MOVQ c_base+96(FP), SI
    MOVQ (SI)(AX*1), R9            // R9 = &c[phase][0]
    MOVQ d_base+120(FP), SI
    MOVQ (SI)(AX*1), R10           // R10 = &d[phase][0]
    // hist window base = &hist[div]
    MOVQ hist_base+24(FP), SI
    LEAQ (SI)(BX*4), SI            // SI = &hist[div]

    // ---- inner dot body, verbatim from cubicInterpDotAVX (CX = taps) ----
    VXORPS Y0, Y0, Y0              // acc0 (re-zeroed per output)
    VXORPS Y6, Y6, Y6              // acc1
    MOVQ CX, AX
    SHRQ $4, AX                    // AX = taps / 16
    JZ   prcubic_loop8_check

prcubic_loop16:
    VMOVUPS 0(R9), Y1
    VMOVUPS 32(R9), Y2
    VMOVUPS 0(R10), Y5
    VFMADD231PS Y5, Y7, Y1         // Y1 = d*x + c
    VMOVUPS 32(R10), Y5
    VFMADD231PS Y5, Y7, Y2
    VMOVUPS 0(R8), Y5
    VFMADD213PS Y5, Y7, Y1         // Y1 = Y1*x + b
    VMOVUPS 32(R8), Y5
    VFMADD213PS Y5, Y7, Y2
    VMOVUPS 0(DI), Y3
    VFMADD213PS Y3, Y7, Y1         // Y1 = Y1*x + a
    VMOVUPS 32(DI), Y4
    VFMADD213PS Y4, Y7, Y2
    VMOVUPS 0(SI), Y5
    VFMADD231PS Y5, Y1, Y0         // acc0 += hist * coef
    VMOVUPS 32(SI), Y5
    VFMADD231PS Y5, Y2, Y6         // acc1 += hist * coef
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, R8
    ADDQ $64, R9
    ADDQ $64, R10
    DECQ AX
    JNZ  prcubic_loop16
    VADDPS Y6, Y0, Y0

prcubic_loop8_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   prcubic_remainder

prcubic_loop8:
    VMOVUPS (R10), Y1
    VMOVUPS (R9), Y2
    VMOVUPS (R8), Y3
    VMOVUPS (DI), Y4
    VMOVUPS (SI), Y5
    VFMADD231PS Y1, Y7, Y2         // Y2 = d*x + c
    VFMADD231PS Y2, Y7, Y3         // Y3 = (d*x+c)*x + b
    VFMADD231PS Y3, Y7, Y4         // Y4 = coef
    VFMADD231PS Y5, Y4, Y0         // acc += hist * coef
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  prcubic_loop8

prcubic_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0             // X0[0] = sum of 8 lanes
    ANDQ $7, CX
    JZ   prcubic_store

prcubic_scalar:
    VMOVSS (R10), X1
    VMOVSS (R9), X2
    VMOVSS (R8), X3
    VMOVSS (DI), X4
    VMOVSS (SI), X5
    VFMADD231SS X1, X7, X2         // X7 lane0 = x
    VFMADD231SS X2, X7, X3
    VFMADD231SS X3, X7, X4
    VFMADD231SS X5, X4, X0
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, R9
    ADDQ $4, R10
    DECQ CX
    JNZ  prcubic_scalar

prcubic_store:
    MOVQ out_base+0(FP), AX
    VMOVSS X0, (AX)(R12*4)         // out[k] = result

    // ---- advance the state machine ----
    // sFrac = step & fracMask;  frac += sFrac
    MOVQ fracBits+176(FP), CX
    MOVQ $1, SI
    SHLQ CX, SI                    // SI = 1 << fracBits
    MOVQ SI, DI                    // DI = 1 << fracBits (= fracMask+1)
    DECQ SI                        // SI = fracMask
    MOVQ step+152(FP), AX
    ANDQ AX, SI                    // SI = step & fracMask = sFrac
    ADDQ SI, R11                   // frac += sFrac
    // if frac > fracMask { frac -= fracMask+1; phase++ }
    MOVQ DI, SI
    DECQ SI                        // SI = fracMask
    CMPQ R11, SI
    JLE  prcubic_nocarry
    SUBQ DI, R11                   // frac -= (fracMask+1)
    INCQ DX                        // phase++
prcubic_nocarry:
    // sPhase = (step>>fracBits) - sDiv*numPhases;  phase += sPhase;  div += sDiv
    MOVQ step+152(FP), AX
    SARQ CX, AX                    // AX = sFull
    MOVQ numPhases+160(FP), SI     // SI = numPhases
    MOVQ R13, DI
    IMULQ SI, DI                   // DI = sDiv*numPhases
    SUBQ DI, AX                    // AX = sPhase
    ADDQ AX, DX                    // phase += sPhase
    ADDQ R13, BX                   // div += sDiv
    // if phase >= numPhases { phase -= numPhases; div++ }
    CMPQ DX, SI
    JLT  prcubic_nonorm
    SUBQ SI, DX
    INCQ BX
prcubic_nonorm:
    INCQ R12                       // k++
    JMP  prcubic_loop

prcubic_done:
    MOVQ R12, ret+184(FP)
    VZEROUPPER
    RET

// func resampleCubic64AVX(out, hist []float64, a, b, c, d [][]float64,
//                                at, step int64, numPhases, tapsPerPhase, fracBits int) int
//
// Fused polyphase cubic resampler (float64): runs the whole output block in one
// pass, reusing cubicInterpDotAVX's inner dot body for each output. Returns the
// number of outputs written.
//
// The inner loop16 here uses per-pointer advances instead of the shared DX
// byte-offset the standalone kernel uses, so DX is free to hold phase; the loads,
// the four accumulators (Y0/Y11/Y12/Y13) and their combine order are unchanged,
// so each output dot is bit-identical to cubicInterpDotAVX at the same tap count.
// sPhase and sFrac are re-derived from step per output with a multiply, not a
// divide. x is splat AVX1-clean (VMOVDDUP + VINSERTF128, no register-source
// VBROADCASTSD). Reserved registers R14/R15/BP are never touched.
//
// Persistent registers across the whole loop:
//   BX  = div
//   DX  = phase
//   R11 = frac
//   R12 = k (output index)
//   R13 = sDiv
//   X8  = fracScale (2^-fracBits, scalar)
//
// Frame layout (6 slices + 2 int64 + 3 int + 1 return):
//   out:  base+0,  len+8,  cap+16
//   hist: base+24, len+32, cap+40
//   a:    base+48, len+56, cap+64
//   b:    base+72, len+80, cap+88
//   c:    base+96, len+104, cap+112
//   d:    base+120,len+128, cap+136
//   at:          +144 (int64)
//   step:        +152 (int64)
//   numPhases:   +160 (int)
//   tapsPerPhase:+168 (int)
//   fracBits:    +176 (int)
//   ret:         +184 (int)
TEXT ·resampleCubic64AVX(SB), NOSPLIT, $0-192
    MOVQ fracBits+176(FP), CX      // CX = fracBits
    MOVQ numPhases+160(FP), R8     // R8 = numPhases

    // sDiv = (step >> fracBits) / numPhases  (one division, hoisted)
    MOVQ step+152(FP), AX
    SARQ CX, AX
    XORQ DX, DX
    DIVQ R8                        // AX = sDiv
    MOVQ AX, R13

    // Seed div, phase, frac from at.
    MOVQ at+144(FP), R11
    MOVQ R11, AX
    SARQ CX, AX                    // AX = full = at >> fracBits
    XORQ DX, DX
    DIVQ R8                        // AX = div, DX = phase
    MOVQ AX, BX                    // BX = div
    MOVQ $1, SI
    SHLQ CX, SI
    DECQ SI                        // SI = fracMask
    ANDQ SI, R11                   // R11 = frac

    // fracScale = 2^-fracBits as float64: bits = (1023 - fracBits) << 52.
    MOVQ $1023, AX
    SUBQ CX, AX
    SHLQ $52, AX
    MOVQ AX, X8                    // X8 = fracScale (scalar)

    XORQ R12, R12                  // k = 0

prcubic_loop:
    MOVQ out_len+8(FP), AX
    CMPQ R12, AX
    JGE  prcubic_done
    MOVQ tapsPerPhase+168(FP), CX  // CX = taps (inner dot length)
    MOVQ BX, AX
    ADDQ CX, AX
    MOVQ hist_len+32(FP), SI
    CMPQ AX, SI
    JGT  prcubic_done

    // x = float64(frac) * fracScale, broadcast to Y7 (AVX1-clean).
    VCVTSI2SDQ R11, X7, X7
    VMULSD   X8, X7, X7            // X7 = x (lane 0)
    VMOVDDUP X7, X7               // X7 = [x, x]
    VINSERTF128 $1, X7, Y7, Y7    // Y7 = [x x x x]

    // Row pointers: byte offset = phase*24.
    LEAQ (DX)(DX*2), AX
    SHLQ $3, AX                    // AX = 24*phase
    MOVQ a_base+48(FP), SI
    MOVQ (SI)(AX*1), DI            // &a[phase][0]
    MOVQ b_base+72(FP), SI
    MOVQ (SI)(AX*1), R8            // &b[phase][0]
    MOVQ c_base+96(FP), SI
    MOVQ (SI)(AX*1), R9            // &c[phase][0]
    MOVQ d_base+120(FP), SI
    MOVQ (SI)(AX*1), R10           // &d[phase][0]
    MOVQ hist_base+24(FP), SI
    LEAQ (SI)(BX*8), SI            // &hist[div]

    // ---- inner dot body (CX = taps); loop16 uses per-pointer advances ----
    VXORPD Y0, Y0, Y0
    VXORPD Y11, Y11, Y11
    VXORPD Y12, Y12, Y12
    VXORPD Y13, Y13, Y13
    MOVQ CX, AX
    SHRQ $4, AX                    // AX = taps / 16
    JZ   prcubic_loop4_check

prcubic_loop16:
    VMOVUPD 0(R9), Y1
    VMOVUPD 0(R10), Y5
    VFMADD231PD Y5, Y7, Y1
    VMOVUPD 32(R9), Y2
    VMOVUPD 32(R10), Y5
    VFMADD231PD Y5, Y7, Y2
    VMOVUPD 64(R9), Y3
    VMOVUPD 64(R10), Y5
    VFMADD231PD Y5, Y7, Y3
    VMOVUPD 96(R9), Y4
    VMOVUPD 96(R10), Y5
    VFMADD231PD Y5, Y7, Y4
    VMOVUPD 0(R8), Y5
    VFMADD213PD Y5, Y7, Y1
    VMOVUPD 32(R8), Y5
    VFMADD213PD Y5, Y7, Y2
    VMOVUPD 64(R8), Y5
    VFMADD213PD Y5, Y7, Y3
    VMOVUPD 96(R8), Y5
    VFMADD213PD Y5, Y7, Y4
    VMOVUPD 0(DI), Y5
    VFMADD213PD Y5, Y7, Y1
    VMOVUPD 32(DI), Y5
    VFMADD213PD Y5, Y7, Y2
    VMOVUPD 64(DI), Y5
    VFMADD213PD Y5, Y7, Y3
    VMOVUPD 96(DI), Y5
    VFMADD213PD Y5, Y7, Y4
    VMOVUPD 0(SI), Y5
    VFMADD231PD Y5, Y1, Y0
    VMOVUPD 32(SI), Y5
    VFMADD231PD Y5, Y2, Y11
    VMOVUPD 64(SI), Y5
    VFMADD231PD Y5, Y3, Y12
    VMOVUPD 96(SI), Y5
    VFMADD231PD Y5, Y4, Y13
    ADDQ $128, SI
    ADDQ $128, DI
    ADDQ $128, R8
    ADDQ $128, R9
    ADDQ $128, R10
    DECQ AX
    JNZ  prcubic_loop16
    VADDPD Y11, Y0, Y0
    VADDPD Y12, Y0, Y0
    VADDPD Y13, Y0, Y0

prcubic_loop4_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   prcubic_remainder

prcubic_loop4:
    VMOVUPD (R10), Y1
    VMOVUPD (R9), Y2
    VMOVUPD (R8), Y3
    VMOVUPD (DI), Y4
    VMOVUPD (SI), Y5
    VFMADD231PD Y1, Y7, Y2
    VFMADD231PD Y2, Y7, Y3
    VFMADD231PD Y3, Y7, Y4
    VFMADD231PD Y5, Y4, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  prcubic_loop4

prcubic_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0             // X0[0] = sum of 4 lanes
    ANDQ $3, CX
    JZ   prcubic_store

prcubic_scalar:
    VMOVSD (R10), X1
    VMOVSD (R9), X2
    VMOVSD (R8), X3
    VMOVSD (DI), X4
    VMOVSD (SI), X5
    VFMADD231SD X1, X7, X2         // X7 lane0 = x
    VFMADD231SD X2, X7, X3
    VFMADD231SD X3, X7, X4
    VFMADD231SD X5, X4, X0
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    DECQ CX
    JNZ  prcubic_scalar

prcubic_store:
    MOVQ out_base+0(FP), AX
    VMOVSD X0, (AX)(R12*8)         // out[k] = result

    // ---- advance the state machine ----
    MOVQ fracBits+176(FP), CX
    MOVQ $1, SI
    SHLQ CX, SI
    MOVQ SI, DI                    // DI = 1 << fracBits (= fracMask+1)
    DECQ SI                        // SI = fracMask
    MOVQ step+152(FP), AX
    ANDQ AX, SI                    // SI = sFrac
    ADDQ SI, R11                   // frac += sFrac
    MOVQ DI, SI
    DECQ SI                        // SI = fracMask
    CMPQ R11, SI
    JLE  prcubic_nocarry
    SUBQ DI, R11                   // frac -= (fracMask+1)
    INCQ DX                        // phase++
prcubic_nocarry:
    MOVQ step+152(FP), AX
    SARQ CX, AX                    // AX = sFull
    MOVQ numPhases+160(FP), SI
    MOVQ R13, DI
    IMULQ SI, DI                   // DI = sDiv*numPhases
    SUBQ DI, AX                    // AX = sPhase
    ADDQ AX, DX                    // phase += sPhase
    ADDQ R13, BX                   // div += sDiv
    CMPQ DX, SI
    JLT  prcubic_nonorm
    SUBQ SI, DX
    INCQ BX
prcubic_nonorm:
    INCQ R12
    JMP  prcubic_loop

prcubic_done:
    MOVQ R12, ret+184(FP)
    VZEROUPPER
    RET
