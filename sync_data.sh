#!/usr/bin/env bash
# sync_data.sh — Copy all VesselReID images into ShipReID-2400 dataset directory.
#
# Usage:
#   bash sync_data.sh
#
# This maps VesselReID normal_split to the 4 folders ShipReID-2400 expects:
#   normal_split/train   -> data/VesselReID/bounding_box_train
#   normal_split/query   -> data/VesselReID/val_query  (used as validation)
#   normal_split/query   -> data/VesselReID/test_query (same query used for test)
#   normal_split/gallery -> data/VesselReID/bounding_box_test
#
# The VesselReID filename format (0002_c1s1_000001_00.jpg) is compatible with
# ShipReID-2400's regex parser: vessel IDs (2-1050) and camera IDs (1-3) are
# both within the expected ranges.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VESSEL_ROOT="$SCRIPT_DIR/VesselReID Dataset/normal_split"
SHIP_ROOT="$SCRIPT_DIR/ShipReID-2400/data/VesselReID"

echo "Source: $VESSEL_ROOT"
echo "Destination: $SHIP_ROOT"
echo ""

# Create destination directories
mkdir -p "$SHIP_ROOT/bounding_box_train"
mkdir -p "$SHIP_ROOT/val_query"
mkdir -p "$SHIP_ROOT/test_query"
mkdir -p "$SHIP_ROOT/bounding_box_test"

echo "Copying bounding_box_train (train split)..."
cp "$VESSEL_ROOT/train/"*.jpg "$SHIP_ROOT/bounding_box_train/"
echo "  Done: $(ls "$SHIP_ROOT/bounding_box_train" | wc -l) files"

echo "Copying val_query (query split)..."
cp "$VESSEL_ROOT/query/"*.jpg "$SHIP_ROOT/val_query/"
echo "  Done: $(ls "$SHIP_ROOT/val_query" | wc -l) files"

echo "Copying test_query (query split — same as val_query for VesselReID)..."
cp "$VESSEL_ROOT/query/"*.jpg "$SHIP_ROOT/test_query/"
echo "  Done: $(ls "$SHIP_ROOT/test_query" | wc -l) files"

echo "Copying bounding_box_test (gallery split)..."
cp "$VESSEL_ROOT/gallery/"*.jpg "$SHIP_ROOT/bounding_box_test/"
echo "  Done: $(ls "$SHIP_ROOT/bounding_box_test" | wc -l) files"

echo ""
echo "Sync complete. Dataset ready at: $SHIP_ROOT"
echo ""
echo "To train, run from ShipReID-2400/:"
echo "  python train.py --config_file configs/ship/vit_base.yml \\"
echo "    DATASETS.ROOT_DIR \"$SCRIPT_DIR/ShipReID-2400/data\" \\"
echo "    DATASETS.NAMES \"('VesselReID')\" \\"
echo "    OUTPUT_DIR ./logs/vessel_full"
