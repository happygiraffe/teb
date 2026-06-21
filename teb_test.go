package teb

import (
	"math/rand/v2"
	"slices"
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
	bm := NewBitmap()
	expected := make(map[uint64]bool)

	// Generate sparse random bits across blocks 0, 1, and 2
	for range 500 {
		idx := rand.N[uint64](1000000)
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
	for b.Loop() {
		idx := rand.N[uint64](1000000)
		bm.Set(idx, true)
	}
}

func BenchmarkGetMutable(b *testing.B) {
	bm := NewBitmap()
	// Set 5% density
	for range 50000 {
		idx := rand.N[uint64](1000000)
		bm.Set(idx, true)
	}

	for b.Loop() {
		idx := rand.N[uint64](1000000)
		bm.Get(idx)
	}
}

func BenchmarkGetSuccinct(b *testing.B) {
	bm := NewBitmap()
	for range 50000 {
		idx := rand.N[uint64](1000000)
		bm.Set(idx, true)
	}

	// Compress to succinct
	bm.Compress()

	for b.Loop() {
		idx := rand.N[uint64](1000000)
		bm.Get(idx)
	}
}

func BenchmarkCompressBlock(b *testing.B) {
	// Prepare a single 64KB block with sparse and clustered bits
	mutable := make([]uint64, 1024)
	// Set some bits
	for range 1000 {
		offset := rand.N[int](65536)
		mutable[offset/64] |= (uint64(1) << (offset % 64))
	}

	for b.Loop() {
		NewSuccinctBlock(mutable)
	}
}

func TestForwardIterator(t *testing.T) {
	bm := NewBitmap()
	setBits := []uint64{5, 12, 100, 1023, 4096, 65000, 65536, 65536 + 10, 120000}
	for _, bit := range setBits {
		bm.Set(bit, true)
	}

	// 1. Test mutable state
	var collected []uint64
	for bit := range bm.All() {
		collected = append(collected, bit)
	}
	if !slices.Equal(collected, setBits) {
		t.Errorf("expected forward iterator to yield %v, got %v", setBits, collected)
	}

	// 2. Test succinct state
	bm.Compress()
	collected = nil
	for bit := range bm.All() {
		collected = append(collected, bit)
	}
	if !slices.Equal(collected, setBits) {
		t.Errorf("expected succinct forward iterator to yield %v, got %v", setBits, collected)
	}
}

func TestBackwardIterator(t *testing.T) {
	bm := NewBitmap()
	setBits := []uint64{5, 12, 100, 1023, 4096, 65000, 65536, 65536 + 10, 120000}
	for _, bit := range setBits {
		bm.Set(bit, true)
	}

	expectedRev := make([]uint64, len(setBits))
	for i, v := range setBits {
		expectedRev[len(setBits)-1-i] = v
	}

	// 1. Test mutable state
	var collected []uint64
	for bit := range bm.AllReverse() {
		collected = append(collected, bit)
	}
	if !slices.Equal(collected, expectedRev) {
		t.Errorf("expected backward iterator to yield %v, got %v", expectedRev, collected)
	}

	// 2. Test succinct state
	bm.Compress()
	collected = nil
	for bit := range bm.AllReverse() {
		collected = append(collected, bit)
	}
	if !slices.Equal(collected, expectedRev) {
		t.Errorf("expected succinct backward iterator to yield %v, got %v", expectedRev, collected)
	}
}

func TestIteratorBreak(t *testing.T) {
	bm := NewBitmap()
	for i := range uint64(100) {
		bm.Set(i*100, true)
	}

	// Break after 5 elements in Forward
	var count int
	for bit := range bm.All() {
		count++
		if count == 5 {
			_ = bit
			break
		}
	}
	if count != 5 {
		t.Errorf("expected to yield exactly 5 elements before breaking, got %d", count)
	}

	// Compress and test break
	bm.Compress()
	count = 0
	for bit := range bm.All() {
		count++
		if count == 5 {
			_ = bit
			break
		}
	}
	if count != 5 {
		t.Errorf("expected compressed to yield exactly 5 elements before breaking, got %d", count)
	}

	// Break after 5 elements in Reverse (mutable)
	bm2 := NewBitmap()
	for i := range uint64(100) {
		bm2.Set(i*100, true)
	}
	count = 0
	for bit := range bm2.AllReverse() {
		count++
		if count == 5 {
			_ = bit
			break
		}
	}
	if count != 5 {
		t.Errorf("expected reverse to yield exactly 5 elements before breaking, got %d", count)
	}

	// Break after 5 elements in Reverse (succinct)
	bm2.Compress()
	count = 0
	for bit := range bm2.AllReverse() {
		count++
		if count == 5 {
			_ = bit
			break
		}
	}
	if count != 5 {
		t.Errorf("expected compressed reverse to yield exactly 5 elements before breaking, got %d", count)
	}
}

func TestEmptyAndFullIterator(t *testing.T) {
	// Empty bitmap
	bmEmpty := NewBitmap()
	for bit := range bmEmpty.All() {
		t.Errorf("expected no elements from empty bitmap, got %d", bit)
	}
	for bit := range bmEmpty.AllReverse() {
		t.Errorf("expected no elements from empty bitmap, got %d", bit)
	}

	// Full block bitmap (mutable)
	bmFull := NewBitmap()
	for i := range uint64(65536) {
		bmFull.Set(i, true)
	}
	var count int
	for range bmFull.All() {
		count++
	}
	if count != 65536 {
		t.Errorf("expected 65536 elements, got %d", count)
	}

	// Full block bitmap (succinct)
	bmFull.Compress()
	count = 0
	for range bmFull.All() {
		count++
	}
	if count != 65536 {
		t.Errorf("expected 65536 elements in succinct, got %d", count)
	}
}

func BenchmarkIteratorForward(b *testing.B) {
	bm := NewBitmap()
	// Set 5% density
	for range 50000 {
		idx := rand.N[uint64](1000000)
		bm.Set(idx, true)
	}
	bm.Compress()

	b.ResetTimer()
	for b.Loop() {
		for bit := range bm.All() {
			_ = bit
		}
	}
}

func BenchmarkIteratorBackward(b *testing.B) {
	bm := NewBitmap()
	// Set 5% density
	for range 50000 {
		idx := rand.N[uint64](1000000)
		bm.Set(idx, true)
	}
	bm.Compress()

	b.ResetTimer()
	for b.Loop() {
		for bit := range bm.AllReverse() {
			_ = bit
		}
	}
}
