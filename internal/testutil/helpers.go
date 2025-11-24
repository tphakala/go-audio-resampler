// Package testutil provides reusable test helper functions for audio resampler tests.
package testutil

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Default tolerances for various test scenarios.
const (
	DefaultTolerance   = 1e-10
	MagnitudeTolerance = 1e-2
	WindowTolerance    = 1e-10
	DBTolerance        = 0.01
)

// halfDivisor is used for finding center indices in symmetric arrays.
const halfDivisor = 2

// AssertSymmetric verifies that a slice is symmetric (s[i] == s[n-1-i]).
func AssertSymmetric(t *testing.T, s []float64, tolerance float64, msgAndArgs ...any) bool {
	t.Helper()
	n := len(s)
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		if !assert.InDelta(t, s[i], s[j], tolerance,
			"slice not symmetric at i=%d: s[%d]=%f != s[%d]=%f", i, i, s[i], j, s[j]) {
			return false
		}
	}
	return true
}

// AssertNoNaNOrInf verifies that no elements in the slice are NaN or Inf.
func AssertNoNaNOrInf(t *testing.T, s []float64, msgAndArgs ...any) bool {
	t.Helper()
	for i, v := range s {
		if math.IsNaN(v) {
			return assert.Fail(t, "found NaN", "s[%d] is NaN", i)
		}
		if math.IsInf(v, 0) {
			return assert.Fail(t, "found Inf", "s[%d] is Inf", i)
		}
	}
	return true
}

// AssertAllInRange verifies that all elements are within [min, max].
func AssertAllInRange(t *testing.T, s []float64, minVal, maxVal float64, msgAndArgs ...any) bool {
	t.Helper()
	for i, v := range s {
		if v < minVal || v > maxVal {
			return assert.Fail(t, "value out of range",
				"s[%d]=%f is outside range [%f, %f]", i, v, minVal, maxVal)
		}
	}
	return true
}

// AssertDCGain verifies that the sum of coefficients equals the expected DC gain.
func AssertDCGain(t *testing.T, coeffs []float64, expectedGain, tolerance float64) bool {
	t.Helper()
	var sum float64
	for _, c := range coeffs {
		sum += c
	}
	return assert.InDelta(t, expectedGain, sum, tolerance,
		"DC gain = %f, want %f", sum, expectedGain)
}

// AssertMonotonic verifies that a slice is monotonically increasing.
func AssertMonotonic(t *testing.T, s []float64, msgAndArgs ...any) bool {
	t.Helper()
	for i := 1; i < len(s); i++ {
		if s[i] < s[i-1] {
			return assert.Fail(t, "not monotonic",
				"s[%d]=%f < s[%d]=%f", i, s[i], i-1, s[i-1])
		}
	}
	return true
}

// AssertCenterIsMax verifies that the center element is the maximum value.
func AssertCenterIsMax(t *testing.T, s []float64, msgAndArgs ...any) bool {
	t.Helper()
	if len(s) == 0 {
		return assert.Fail(t, "empty slice")
	}
	centerIdx := len(s) / halfDivisor
	centerValue := s[centerIdx]
	for i, v := range s {
		if v > centerValue {
			return assert.Fail(t, "center is not max",
				"s[%d]=%f > center s[%d]=%f", i, v, centerIdx, centerValue)
		}
	}
	return true
}

// AssertRelativeError verifies that the relative error between actual and expected is within tolerance.
func AssertRelativeError(t *testing.T, expected, actual, tolerance float64, msgAndArgs ...any) bool {
	t.Helper()
	if expected == 0 {
		return assert.InDelta(t, expected, actual, tolerance, msgAndArgs...)
	}
	relError := math.Abs(actual-expected) / math.Abs(expected)
	return assert.LessOrEqual(t, relError, tolerance,
		"relative error %e exceeds tolerance %e (expected=%f, actual=%f)",
		relError, tolerance, expected, actual)
}

// AssertLengthEquals verifies that a slice has the expected length.
func AssertLengthEquals(t *testing.T, s []float64, expectedLen int, msgAndArgs ...any) bool {
	t.Helper()
	return assert.Len(t, s, expectedLen, msgAndArgs...)
}

// AssertOddLength verifies that a slice has an odd length.
func AssertOddLength(t *testing.T, s []float64, msgAndArgs ...any) bool {
	t.Helper()
	return assert.Equal(t, 1, len(s)%halfDivisor, "slice length %d is not odd", len(s))
}

// AssertInRange verifies that a value is within [min, max].
func AssertInRange(t *testing.T, value, minVal, maxVal float64, msgAndArgs ...any) bool {
	t.Helper()
	if value < minVal || value > maxVal {
		return assert.Fail(t, "value out of range",
			"value %f is outside range [%f, %f]", value, minVal, maxVal)
	}
	return true
}
