#!/bin/bash
# Signal Alignment Pipeline
# Usage: ./run_signal_alignment.sh <pod5_file> <reference.fasta> <output_prefix>

set -e

# Activate environment
source ~/mambaforge/etc/profile.d/conda.sh
conda activate nanopore

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <pod5_file> <reference.fasta> <output_prefix>"
    echo ""
    echo "Example:"
    echo "  $0 filtered_pod_files/5mc_filtered_adapter_01.pod5 reference.fasta output/aligned"
    exit 1
fi

POD5_FILE="$1"
REFERENCE="$2"
OUTPUT_PREFIX="$3"

# Paths
DORADO=~/software/dorado-1.3.0-linux-x64/bin/dorado
MODEL=~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

echo "=== Step 1: Basecalling with move tables ==="
$DORADO basecaller \
    "$MODEL" \
    "$POD5_FILE" \
    --emit-moves \
    --reference "$REFERENCE" \
    > "${OUTPUT_PREFIX}_with_moves.bam"

echo "=== Step 2: Sort and Index BAM ==="
samtools sort -o "${OUTPUT_PREFIX}_with_moves_sorted.bam" "${OUTPUT_PREFIX}_with_moves.bam"
mv "${OUTPUT_PREFIX}_with_moves_sorted.bam" "${OUTPUT_PREFIX}_with_moves.bam"
samtools index "${OUTPUT_PREFIX}_with_moves.bam"

echo "=== Step 3: Signal alignment with uncalled4 ==="
uncalled4 align \
    --ref "$REFERENCE" \
    --reads "$POD5_FILE" \
    --bam-in "${OUTPUT_PREFIX}_with_moves.bam" \
    -o "${OUTPUT_PREFIX}_signal_aligned_unsorted.bam" \
    --basecaller-profile dna_r10.4.1_400bps \
    -p 8

echo "=== Step 4: Sort and Index final BAM ==="
samtools sort -o "${OUTPUT_PREFIX}_signal_aligned.bam" "${OUTPUT_PREFIX}_signal_aligned_unsorted.bam"
rm "${OUTPUT_PREFIX}_signal_aligned_unsorted.bam"
samtools index "${OUTPUT_PREFIX}_signal_aligned.bam"

echo "=== Step 5: Verification ==="
echo "Checking for mv (move) and ts (stride) tags:"
samtools view "${OUTPUT_PREFIX}_signal_aligned.bam" | head -1 | tr '\t' '\n' | grep -E "^(mv|ts):" || echo "Tags not found in first read"

echo ""
echo "=== Done! ==="
echo "Output files:"
echo "  ${OUTPUT_PREFIX}_with_moves.bam     - BAM with move tables"
echo "  ${OUTPUT_PREFIX}_signal_aligned.bam - Signal-aligned BAM"
