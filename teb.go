package teb

import (
	"math/bits"
	"sync"
)

// Bitmap represents a dynamic, transactional, and space-efficient Tree-Encoded Bitmap.
// It is thread-safe and supports highly efficient real-time mutations and compact storage
// by dividing the bit space into 65,536-bit blocks.
type Bitmap struct {
	mu     sync.RWMutex
	blocks map[uint32]*Block
	length uint64
}

// NewBitmap creates and initializes an empty Bitmap.
func NewBitmap() *Bitmap {
	return &Bitmap{
		blocks: make(map[uint32]*Block),
	}
}

// Get performs a point lookup at the specified 0-indexed bit position.
// It returns true if the bit is set (1) and false otherwise.
func (b *Bitmap) Get(idx uint64) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()

	blockIdx := uint32(idx / 65536)
	offset := uint16(idx % 65536)

	block, exists := b.blocks[blockIdx]
	if !exists {
		return false
	}

	return block.Get(offset)
}

// Set sets the bit at the specified index to the given boolean value.
// It automatically handles dynamic decompression/compaction boundaries.
func (b *Bitmap) Set(idx uint64, val bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	blockIdx := uint32(idx / 65536)
	offset := uint16(idx % 65536)

	block, exists := b.blocks[blockIdx]
	if !exists {
		if !val {
			return // Setting false on non-existent block is a no-op
		}
		block = &Block{
			mutable: make([]uint64, 1024),
		}
		b.blocks[blockIdx] = block
	}

	block.Set(offset, val)

	if val && idx >= b.length {
		b.length = idx + 1
	}
}

// Compress compacts all uncompressed mutable blocks into succinct TEB blocks,
// but only if the compressed TEB representation actually saves space.
// It returns the number of blocks successfully compressed to TEB.
func (b *Bitmap) Compress() int {
	b.mu.Lock()
	defer b.mu.Unlock()

	compressedCount := 0
	for blockIdx, block := range b.blocks {
		if block.mutable != nil {
			if block.Compress() {
				// If block became completely empty, we can delete it
				if block.mutable == nil && block.succinct == nil {
					delete(b.blocks, blockIdx)
				} else {
					compressedCount++
				}
			}
		}
	}
	return compressedCount
}

// Len returns the current logical length of the bitmap (1 + highest set bit index).
func (b *Bitmap) Len() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.length
}

// BlockCount returns the number of active 64KB blocks.
func (b *Bitmap) BlockCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.blocks)
}

// Block represents a single 65,536-bit partition.
// It maintains a hybrid state: either uncompressed mutable (for fast transactional updates)
// or compressed succinct (for compact storage and fast read lookups).
type Block struct {
	mutable  []uint64       // 1024 uint64s (8KB) if in mutable state
	succinct *SuccinctBlock // Compressed tree-encoded bitmap block
}

// Get performs a fast point lookup in the block.
func (block *Block) Get(offset uint16) bool {
	if block.mutable != nil {
		word := offset / 64
		bit := offset % 64
		return (block.mutable[word] & (uint64(1) << bit)) != 0
	}
	if block.succinct != nil {
		return block.succinct.Get(offset)
	}
	return false
}

// Set performs an in-place update. If the block is currently compressed,
// it decompresses it to mutable state first.
func (block *Block) Set(offset uint16, val bool) {
	if block.succinct != nil {
		block.Decompress()
	}

	word := offset / 64
	bit := offset % 64
	if val {
		block.mutable[word] |= (uint64(1) << bit)
	} else {
		block.mutable[word] &= ^(uint64(1) << bit)
	}
}

// Decompress converts the succinct TEB block back to a mutable raw bitmap.
func (block *Block) Decompress() {
	if block.succinct == nil {
		return
	}
	block.mutable = make([]uint64, 1024)
	block.succinct.DecompressTo(block.mutable)
	block.succinct = nil
}

// Compress evaluates the mutable block and encodes it as a SuccinctBlock
// if it reduces memory footprint compared to the 8KB raw bitmap.
func (block *Block) Compress() bool {
	if block.mutable == nil {
		return false
	}

	// 1. Check if the block is entirely empty
	isEmpty := true
	for _, word := range block.mutable {
		if word != 0 {
			isEmpty = false
			break
		}
	}
	if isEmpty {
		block.mutable = nil
		block.succinct = nil
		return true
	}

	// 2. Build the succinct representation
	sb := NewSuccinctBlock(block.mutable)

	// 3. Compute exact physical size of the SuccinctBlock in bytes.
	// Raw bitmap is 1024 * 8 = 8192 bytes.
	sbSize := len(sb.t)*8 + len(sb.l)*8 + len(sb.rankLUT)*4
	if sbSize < 8192 {
		block.succinct = sb
		block.mutable = nil
		return true
	}

	return false
}

// SuccinctBlock represents the static, succinct level-order binary tree-encoded bitmap.
type SuccinctBlock struct {
	t       []uint64 // Tree structure bit vector (1 for inner node, 0 for leaf)
	l       []uint64 // Leaf labels bit vector
	rankLUT []uint32 // precomputed inclusive Rank lookup table for T (512-bit block granularity)
}

// buildNode is a temporary helper node during tree construction and pruning.
type buildNode struct {
	IsLeaf bool
	Label  bool
}

// NewSuccinctBlock constructs a SuccinctBlock from a raw 65,536-bit bitmap.
// It establishes a perfect binary tree of height 16, prunes it bottom-up,
// and serializes it in BFS level-order.
func NewSuccinctBlock(bitmap []uint64) *SuccinctBlock {
	// A perfect binary tree of height 16 has 2^17 - 1 = 131,071 nodes.
	// We use a flat array representation for blazing-fast indexing without pointer dereferencing.
	tree := make([]buildNode, 131071)

	// 1. Initialize leaf nodes at Level 16 (indices 65535 to 131070)
	for k := 0; k < 65536; k++ {
		word := k / 64
		bit := k % 64
		val := (bitmap[word] & (uint64(1) << uint(bit))) != 0
		tree[65535+k] = buildNode{IsLeaf: true, Label: val}
	}

	// 2. Perform bottom-up pruning
	// For each depth from 15 down to 0, prune siblings if they are leaves and share the same label.
	for d := 15; d >= 0; d-- {
		start := (1 << uint(d)) - 1
		end := (1 << uint(d+1)) - 2
		for i := start; i <= end; i++ {
			left := 2*i + 1
			right := 2*i + 2
			if tree[left].IsLeaf && tree[right].IsLeaf && tree[left].Label == tree[right].Label {
				tree[i] = buildNode{IsLeaf: true, Label: tree[left].Label}
			} else {
				tree[i] = buildNode{IsLeaf: false}
			}
		}
	}

	// 3. Serialize tree using BFS level-order traversal
	queue := make([]uint32, 0, 131071)
	queue = append(queue, 0) // Push root

	var tBits []uint64
	var lBits []uint64
	var tCount uint32
	var lCount uint32

	appendT := func(val bool) {
		word := tCount / 64
		bit := tCount % 64
		if word >= uint32(len(tBits)) {
			tBits = append(tBits, 0)
		}
		if val {
			tBits[word] |= (uint64(1) << bit)
		}
		tCount++
	}

	appendL := func(val bool) {
		word := lCount / 64
		bit := lCount % 64
		if word >= uint32(len(lBits)) {
			lBits = append(lBits, 0)
		}
		if val {
			lBits[word] |= (uint64(1) << bit)
		}
		lCount++
	}

	head := 0
	for head < len(queue) {
		i := queue[head]
		head++

		if tree[i].IsLeaf {
			appendT(false) // 0 bit for Leaf in T
			appendL(tree[i].Label)
		} else {
			appendT(true) // 1 bit for Inner Node in T
			queue = append(queue, uint32(2*i+1), uint32(2*i+2))
		}
	}

	// 4. Precompute the Rank Lookup Table (LuT)
	// Resolution is 512 bits (8 uint64 words).
	numBlocks := (tCount + 511) / 512
	rankLUT := make([]uint32, numBlocks)
	currentRank := uint32(0)
	for b := uint32(0); b < numBlocks; b++ {
		rankLUT[b] = currentRank
		wordStart := b * 8
		for w := uint32(0); w < 8; w++ {
			idx := wordStart + w
			if idx < uint32(len(tBits)) {
				currentRank += uint32(bits.OnesCount64(tBits[idx]))
			}
		}
	}

	return &SuccinctBlock{
		t:       tBits,
		l:       lBits,
		rankLUT: rankLUT,
	}
}

// Get performs a succinct random point lookup in O(log N) time.
func (sb *SuccinctBlock) Get(offset uint16) bool {
	i := uint32(0)  // Start at root
	j := uint32(15) // Binary path selector (height 16 tree)

	for sb.isInnerNode(i) {
		direction := uint32((offset >> j) & 1)
		i = sb.leftChild(i) + direction
		j--
	}

	return sb.label(i)
}

// Rank calculates the inclusive rank (number of 1s in T[0..i]).
// Uses the Rank Lookup Table and native OnesCount64 assembly instructions.
func (sb *SuccinctBlock) Rank(i uint32) uint32 {
	block := i / 512
	bitCount := (i % 512) + 1

	r := sb.rankLUT[block]
	wordStart := block * 8

	fullWords := bitCount / 64
	for w := uint32(0); w < fullWords; w++ {
		r += uint32(bits.OnesCount64(sb.t[wordStart+w]))
	}

	remainingBits := bitCount % 64
	if remainingBits > 0 {
		mask := (uint64(1) << remainingBits) - 1
		r += uint32(bits.OnesCount64(sb.t[wordStart+fullWords] & mask))
	}

	return r
}

func (sb *SuccinctBlock) isInnerNode(i uint32) bool {
	word := i / 64
	bit := i % 64
	if word >= uint32(len(sb.t)) {
		return false
	}
	return (sb.t[word] & (uint64(1) << bit)) != 0
}

func (sb *SuccinctBlock) leftChild(i uint32) uint32 {
	return 2*sb.Rank(i) - 1
}

func (sb *SuccinctBlock) label(i uint32) bool {
	labelIdx := i - sb.Rank(i)
	word := labelIdx / 64
	bit := labelIdx % 64
	if word >= uint32(len(sb.l)) {
		return false
	}
	return (sb.l[word] & (uint64(1) << bit)) != 0
}

// DecompressTo reconstructs the raw bitmap from the succinct tree structure
// in O(M) time where M is the number of nodes in the pruned tree.
func (sb *SuccinctBlock) DecompressTo(bitmap []uint64) {
	var recurse func(i uint32, begin uint32, length uint32)
	recurse = func(i uint32, begin uint32, length uint32) {
		if !sb.isInnerNode(i) {
			if sb.label(i) {
				// Set the range [begin, begin + length) in the bitmap to 1
				for k := uint32(0); k < length; k++ {
					idx := begin + k
					bitmap[idx/64] |= (uint64(1) << (idx % 64))
				}
			}
			return
		}

		leftChild := sb.leftChild(i)
		half := length / 2
		recurse(leftChild, begin, half)
		recurse(leftChild+1, begin+half, half)
	}

	recurse(0, 0, 65536)
}
