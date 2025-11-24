package main

// Default command-line flag values
const (
	defaultInputRate  = 44100.0 // CD quality sample rate
	defaultOutputRate = 48000.0 // DAT/DVD sample rate
	defaultChannels   = 2       // Stereo
)

// Test signal parameters
const (
	testSignalFrequency = 1000.0 // 1 kHz test tone
	testSignalSamples   = 1000   // Default test signal length
)

// Demo sample rates for testing
const (
	sampleRateCD    = 44100.0 // CD quality
	sampleRateDAT   = 48000.0 // DAT/DVD
	sampleRate2xCD  = 88200.0 // 2x CD
	sampleRateHiRes = 96000.0 // Hi-res audio
)

// Demo channel configurations
const (
	monoChannels   = 1
	stereoChannels = 2
	surround5_1    = 6
	surround7_1    = 8
)

// Memory conversion
const (
	bytesPerKilobyte = 1024
)
