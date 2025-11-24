package pipeline

import (
	"sync"
)

// RingBuffer implements a circular buffer for audio samples.
// It's designed for efficient streaming between pipeline stages.
type RingBuffer struct {
	data     []float64
	capacity int
	size     int
	readPos  int
	writePos int
	mu       sync.Mutex
}

// NewRingBuffer creates a new ring buffer with the specified capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	if capacity < 1 {
		capacity = 1
	}

	return &RingBuffer{
		data:     make([]float64, capacity),
		capacity: capacity,
		size:     0,
		readPos:  0,
		writePos: 0,
	}
}

// Write adds samples to the buffer.
// If the buffer doesn't have enough space, it will grow automatically.
func (b *RingBuffer) Write(samples []float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	needed := len(samples)
	if needed == 0 {
		return
	}

	// Grow buffer if needed
	if b.size+needed > b.capacity {
		b.grow(b.size + needed)
	}

	// Write samples (may wrap around)
	for _, sample := range samples {
		b.data[b.writePos] = sample
		b.writePos = (b.writePos + 1) % b.capacity
		b.size++
	}
}

// Read retrieves up to n samples from the buffer.
// Returns fewer samples if less are available.
func (b *RingBuffer) Read(n int) []float64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	if n > b.size {
		n = b.size
	}
	if n <= 0 {
		return []float64{}
	}

	result := make([]float64, n)

	// Read samples (may wrap around)
	for i := 0; i < n; i++ {
		result[i] = b.data[b.readPos]
		b.readPos = (b.readPos + 1) % b.capacity
		b.size--
	}

	return result
}

// Peek returns up to n samples without removing them from the buffer.
func (b *RingBuffer) Peek(n int) []float64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	if n > b.size {
		n = b.size
	}
	if n <= 0 {
		return []float64{}
	}

	result := make([]float64, n)
	readPos := b.readPos

	// Copy samples without modifying buffer state
	for i := 0; i < n; i++ {
		result[i] = b.data[readPos]
		readPos = (readPos + 1) % b.capacity
	}

	return result
}

// ReadAll retrieves all available samples from the buffer.
func (b *RingBuffer) ReadAll() []float64 {
	return b.Read(b.size)
}

// Available returns the number of samples available for reading.
func (b *RingBuffer) Available() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.size
}

// Space returns the available space for writing.
func (b *RingBuffer) Space() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.capacity - b.size
}

// Capacity returns the current buffer capacity.
func (b *RingBuffer) Capacity() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.capacity
}

// Clear removes all samples from the buffer.
func (b *RingBuffer) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.size = 0
	b.readPos = 0
	b.writePos = 0
}

// grow increases the buffer capacity to at least the specified size.
func (b *RingBuffer) grow(minCapacity int) {
	// Calculate new capacity (double until sufficient)
	newCapacity := b.capacity
	for newCapacity < minCapacity {
		newCapacity *= 2
	}

	// Allocate new buffer
	newData := make([]float64, newCapacity)

	// Copy existing data to maintain order
	if b.size > 0 {
		if b.readPos < b.writePos {
			// Data is contiguous
			copy(newData, b.data[b.readPos:b.writePos])
		} else {
			// Data wraps around
			n1 := copy(newData, b.data[b.readPos:])
			copy(newData[n1:], b.data[:b.writePos])
		}
	}

	// Update buffer state
	b.data = newData
	b.capacity = newCapacity
	b.readPos = 0
	b.writePos = b.size
}

// FIFOBuffer is an alternative implementation optimized for FIFO operations.
// It avoids the overhead of modulo operations by using power-of-2 sizes.
type FIFOBuffer struct {
	data     []float64
	mask     int // Capacity - 1 (for bitwise AND instead of modulo)
	size     int
	readPos  uint32
	writePos uint32
	mu       sync.Mutex
}

// NewFIFOBuffer creates a new FIFO buffer.
// Capacity is rounded up to the nearest power of 2.
func NewFIFOBuffer(capacity int) *FIFOBuffer {
	// Round up to power of 2
	cap2 := 1
	for cap2 < capacity {
		cap2 <<= 1
	}

	return &FIFOBuffer{
		data: make([]float64, cap2),
		mask: cap2 - 1,
	}
}

// Write adds samples to the FIFO buffer.
func (f *FIFOBuffer) Write(samples []float64) {
	f.mu.Lock()
	defer f.mu.Unlock()

	for _, sample := range samples {
		if f.size >= len(f.data) {
			// Buffer full, grow
			f.grow()
		}

		f.data[f.writePos&uint32(f.mask)] = sample
		f.writePos++
		f.size++
	}
}

// Read retrieves up to n samples from the FIFO buffer.
func (f *FIFOBuffer) Read(n int) []float64 {
	f.mu.Lock()
	defer f.mu.Unlock()

	if n > f.size {
		n = f.size
	}
	if n <= 0 {
		return []float64{}
	}

	result := make([]float64, n)

	for i := 0; i < n; i++ {
		result[i] = f.data[f.readPos&uint32(f.mask)]
		f.readPos++
		f.size--
	}

	return result
}

// grow doubles the FIFO buffer capacity.
func (f *FIFOBuffer) grow() {
	newCap := len(f.data) * bufferGrowthFactor
	newData := make([]float64, newCap)

	// Copy existing data
	for i := 0; i < f.size; i++ {
		newData[i] = f.data[(f.readPos+uint32(i))&uint32(f.mask)]
	}

	f.data = newData
	f.mask = newCap - 1
	f.readPos = 0
	f.writePos = uint32(f.size)
}
