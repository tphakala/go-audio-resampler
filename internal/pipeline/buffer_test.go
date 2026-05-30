// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package pipeline

import "testing"

func assertFloat64SliceEqual(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("value mismatch at %d: got=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestRingBufferReadIntoContiguous(t *testing.T) {
	b := NewRingBuffer(8)
	b.Write([]float64{1, 2, 3, 4})

	dst := make([]float64, 3)
	n := b.ReadInto(dst)
	if n != 3 {
		t.Fatalf("unexpected read count: got=%d want=3", n)
	}
	assertFloat64SliceEqual(t, dst, []float64{1, 2, 3})
	if b.Available() != 1 {
		t.Fatalf("unexpected remaining samples: got=%d want=1", b.Available())
	}
}

func TestRingBufferReadIntoWrapAround(t *testing.T) {
	b := NewRingBuffer(4)
	b.Write([]float64{1, 2, 3})

	tmp := make([]float64, 2)
	n := b.ReadInto(tmp)
	if n != 2 {
		t.Fatalf("unexpected initial read count: got=%d want=2", n)
	}
	assertFloat64SliceEqual(t, tmp, []float64{1, 2})

	b.Write([]float64{4, 5, 6}) // force wrap-around layout

	dst := make([]float64, 4)
	n = b.ReadInto(dst)
	if n != 4 {
		t.Fatalf("unexpected wrap read count: got=%d want=4", n)
	}
	assertFloat64SliceEqual(t, dst, []float64{3, 4, 5, 6})
	if b.Available() != 0 {
		t.Fatalf("expected empty buffer after drain, got=%d", b.Available())
	}
}

func TestRingBufferReadIntoShortDestination(t *testing.T) {
	b := NewRingBuffer(8)
	b.Write([]float64{10, 20, 30})

	dst := make([]float64, 2)
	n := b.ReadInto(dst)
	if n != 2 {
		t.Fatalf("unexpected read count: got=%d want=2", n)
	}
	assertFloat64SliceEqual(t, dst, []float64{10, 20})
	if b.Available() != 1 {
		t.Fatalf("unexpected remaining samples: got=%d want=1", b.Available())
	}
}

func TestRingBufferReadIntoEmptyDestination(t *testing.T) {
	b := NewRingBuffer(8)
	b.Write([]float64{1, 2})

	dst := make([]float64, 0)
	n := b.ReadInto(dst)
	if n != 0 {
		t.Fatalf("unexpected read count: got=%d want=0", n)
	}
	if b.Available() != 2 {
		t.Fatalf("buffer should be unchanged, got available=%d want=2", b.Available())
	}
}

func TestRingBufferReadIntoFullDrain(t *testing.T) {
	b := NewRingBuffer(8)
	b.Write([]float64{7, 8, 9})

	dst := make([]float64, 10)
	n := b.ReadInto(dst)
	if n != 3 {
		t.Fatalf("unexpected read count: got=%d want=3", n)
	}
	assertFloat64SliceEqual(t, dst[:n], []float64{7, 8, 9})
	if b.Available() != 0 {
		t.Fatalf("expected empty buffer after drain, got=%d", b.Available())
	}

	n = b.ReadInto(dst)
	if n != 0 {
		t.Fatalf("expected zero reads on empty buffer, got=%d", n)
	}
}

func TestRingBufferExtended(t *testing.T) {
	// NewRingBuffer(0) clamps capacity to 1
	b0 := NewRingBuffer(0)
	if b0.Capacity() != 1 {
		t.Fatalf("expected capacity to be clamped to 1, got %d", b0.Capacity())
	}

	b := NewRingBuffer(4)

	// Write empty slice
	b.Write([]float64{})
	if b.Available() != 0 {
		t.Fatalf("expected available to be 0, got %d", b.Available())
	}

	// Read from empty
	rEmpty := b.Read(5)
	assertFloat64SliceEqual(t, rEmpty, []float64{})

	// Capacity, Space, Available
	if b.Capacity() != 4 {
		t.Fatalf("expected capacity 4, got %d", b.Capacity())
	}
	if b.Space() != 4 {
		t.Fatalf("expected space 4, got %d", b.Space())
	}

	b.Write([]float64{1, 2})
	if b.Available() != 2 {
		t.Fatalf("expected available 2, got %d", b.Available())
	}
	if b.Space() != 2 {
		t.Fatalf("expected space 2, got %d", b.Space())
	}

	// Read(0)
	assertFloat64SliceEqual(t, b.Read(0), []float64{})

	// Read(n > size)
	assertFloat64SliceEqual(t, b.Read(5), []float64{1, 2})

	// Clear
	b.Write([]float64{3, 4})
	b.Clear()
	if b.Available() != 0 {
		t.Fatalf("expected available 0 after Clear, got %d", b.Available())
	}

	// ReadAll
	b.Write([]float64{5, 6, 7})
	assertFloat64SliceEqual(t, b.ReadAll(), []float64{5, 6, 7})

	// grow - contiguous layout path
	bContig := NewRingBuffer(4)
	bContig.Write([]float64{1, 2})    // readPos = 0, writePos = 2, size = 2
	bContig.Write([]float64{3, 4, 5}) // size+needed = 2+3 = 5 > 4 capacity, triggers grow (contiguous)
	assertFloat64SliceEqual(t, bContig.ReadAll(), []float64{1, 2, 3, 4, 5})

	// grow - wrapped layout path
	bWrapped := NewRingBuffer(4)
	bWrapped.Write([]float64{1, 2, 3}) // readPos = 0, writePos = 3, size = 3
	bWrapped.Read(2)                   // readPos = 2, writePos = 3, size = 1
	bWrapped.Write([]float64{4, 5})    // readPos = 2, writePos = 1, size = 3 (wraps around)
	// Now write more to trigger growth while wrapped (size=3, needed=2, total=5 > 4 capacity)
	bWrapped.Write([]float64{6, 7})
	assertFloat64SliceEqual(t, bWrapped.ReadAll(), []float64{3, 4, 5, 6, 7})
}
