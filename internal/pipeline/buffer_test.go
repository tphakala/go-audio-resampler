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
