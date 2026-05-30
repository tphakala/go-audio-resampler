// SPDX-FileCopyrightText: 2025 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

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
	for i := range n {
		result[i] = b.data[b.readPos]
		b.readPos = (b.readPos + 1) % b.capacity
		b.size--
	}

	return result
}

// ReadInto copies up to len(dst) samples into the caller-provided buffer,
// consuming them from the ring. Returns the number of samples read.
func (b *RingBuffer) ReadInto(dst []float64) int {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := min(len(dst), b.size)
	if n <= 0 {
		return 0
	}

	firstChunk := b.capacity - b.readPos
	if firstChunk >= n {
		copy(dst[:n], b.data[b.readPos:b.readPos+n])
	} else {
		copy(dst[:firstChunk], b.data[b.readPos:b.readPos+firstChunk])
		copy(dst[firstChunk:n], b.data[:n-firstChunk])
	}
	b.readPos = (b.readPos + n) % b.capacity
	b.size -= n
	return n
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
