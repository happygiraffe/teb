package teb

import (
	"math/rand"
	"testing"
)

func TestBasicSetGet(t *testing.T) {
	bm := NewBitmap()

	// Initially, all bits should be false
	for _, idx := range []uint64{0, 1, 100, 65535, 65536, 1234567} {
		if bm.Get(idx) {
			t.Errorf("expected bit %d to be false initially", idx)
		}
	}

	// Set some bits
	bm.Set(0, true)
	bm.Set(65535, true)
	bm.Set(65536, true)
	bm.Set(1000000, true)

	// Verify set bits
	if !bm.Get(0) {
		t.Error("expected bit 0 to be true")
	}
	if !bm.Get(65535) {
		t.Error("expected bit 65535 to be true")
	}
	if !bm.Get(65536) {
		t.Error("expected bit 65536 to be true")
	}
	if !bm.Get(1000000) {
		t.Error("expected bit 1000000 to be true")
	}

	// Verify unset bits in the same blocks are still false
	if bm.Get(1) {
		t.Error("expected bit 1 to be false")
	}
	if bm.Get(65534) {
		t.Error("expected bit 65534 to be false")
	}
	if bm.Get(65537) {
		t.Error("expected bit 65537 to be false")
	}

	// Unset a bit
	bm.Set(65536, false)
	if bm.Get(65536) {
		t.Error("expected bit 65536 to be false after unsetting")
	}
}

func TestCompressionCorrectness(t *testing.T) {
	bm := NewBitmap()

	// Populate bits in block 0
	setIndices := []uint64{5, 12, 100, 1023, 4096, 65000}
	for _, idx := range setIndices {
		bm.Set(idx, true)
	}

	// Also set some bits in block 1
	bm.Set(65536+100, true)
	bm.Set(65536+500, true)

	// Verify mutable state size before compression
	if bm.BlockCount() != 2 {
		t.Fatalf("expected 2 active blocks, got %d", bm.BlockCount())
	}

	// Compress the entire bitmap
	compressed := bm.Compress()
	if compressed != 2 {
		t.Errorf("expected 2 blocks to be compressed, got %d", compressed)
	}

	// Verify all set bits are still true after compression
	for _, idx := range setIndices {
		if !bm.Get(idx) {
			t.Errorf("expected compressed bit %d to be true", idx)
		}
	}
	if !bm.Get(65536 + 100) {
		t.Error("expected compressed bit 65636 to be true")
	}
	if !bm.Get(65536 + 500) {
		t.Error("expected compressed bit 66036 to be true")
	}

	// Verify unset bits are still false
	if bm.Get(0) || bm.Get(6) || bm.Get(65536) || bm.Get(1000000) {
		t.Error("expected unset bits to remain false after compression")
	}

	// Perform a write to verify transparent decompression on-the-fly
	bm.Set(100, false) // unsetting a bit
	if bm.Get(100) {
		t.Error("expected bit 100 to be false after modification")
	}
	// Verify other bits in block 0 are still intact
	if !bm.Get(5) || !bm.Get(65000) {
		t.Error("expected other bits to remain true after transparent decompression")
	}
}

func TestEmptyAndFullBlock(t *testing.T) {
	// 1. Entirely empty block compression
	bm1 := NewBitmap()
	bm1.Set(10, true)
	bm1.Set(10, false) // set back to false, so block 0 is empty but active

	comp1 := bm1.Compress()
	if comp1 != 0 {
		t.Errorf("expected 0 blocks compressed for empty block, got %d", comp1)
	}
	if bm1.BlockCount() != 0 {
		t.Errorf("expected 0 active blocks, got %d", bm1.BlockCount())
	}

	// 2. Entirely full block compression (65,536 bits all set to 1)
	bm2 := NewBitmap()
	for i := range uint64(65536) {
		bm2.Set(i, true)
	}

	comp2 := bm2.Compress()
	if comp2 != 1 {
		t.Errorf("expected 1 block compressed, got %d", comp2)
	}

	// Validate full block has all bits true
	for i := range uint64(65536) {
		if !bm2.Get(i) {
			t.Fatalf("expected bit %d to be true in fully compressed block", i)
		}
	}
}

func TestRandomSparseClusteredCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	bm := NewBitmap()
	expected := make(map[uint64]bool)

	// Generate sparse random bits across blocks 0, 1, and 2
	for range 500 {
		idx := uint64(rng.Intn(65536 * 3))
		bm.Set(idx, true)
		expected[idx] = true
	}

	// Generate a clustered run
	start := uint64(45000)
	runLen := uint64(3000)
	for i := range runLen {
		bm.Set(start+i, true)
		expected[start+i] = true
	}

	// Compress
	bm.Compress()

	// Verify all indices
	for idx := range uint64(65536 * 3) {
		actual := bm.Get(idx)
		exp := expected[idx]
		if actual != exp {
			t.Fatalf("mismatch at index %d: expected %v, got %v", idx, exp, actual)
		}
	}
}

// BENCHMARKS

func BenchmarkSet(b *testing.B) {
	bm := NewBitmap()
	rng := rand.New(rand.NewSource(42))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := uint64(rng.Intn(1000000))
		bm.Set(idx, true)
	}
}

func BenchmarkGetMutable(b *testing.B) {
	bm := NewBitmap()
	// Set 5% density
	rng := rand.New(rand.NewSource(42))
	for range 50000 {
		idx := uint64(rng.Intn(1000000))
		bm.Set(idx, true)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := uint64(rng.Intn(1000000))
		_ = bm.Get(idx)
	}
}

func BenchmarkGetSuccinct(b *testing.B) {
	bm := NewBitmap()
	rng := rand.New(rand.NewSource(42))
	for range 50000 {
		idx := uint64(rng.Intn(1000000))
		bm.Set(idx, true)
	}

	// Compress to succinct
	bm.Compress()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := uint64(rng.Intn(1000000))
		_ = bm.Get(idx)
	}
}

func BenchmarkCompressBlock(b *testing.B) {
	// Prepare a single 64KB block with sparse and clustered bits
	mutable := make([]uint64, 1024)
	rng := rand.New(rand.NewSource(42))
	// Set some bits
	for range 1000 {
		offset := rng.Intn(65536)
		mutable[offset/64] |= (uint64(1) << (offset % 64))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewSuccinctBlock(mutable)
	}
}
