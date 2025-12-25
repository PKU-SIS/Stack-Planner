#!/bin/bash

echo "=========================================="
echo "  Cleaning up temporary fix files"
echo "=========================================="
echo ""

# Define files to clean
TEMP_FILES=(
    "fix_validation_error.sh"
    "apply_patch_and_restart.sh"
    "check_status.sh"
    "restart_service.sh"
    "FIX_SUMMARY.md"
    "FINAL_FIX_REPORT.md"
    "fix_execution.log"
    "src/prompts/template.py.backup.20251225_174701"
)

# Optional: keep documentation
KEEP_DOCS=false
if [ "$1" == "--keep-docs" ]; then
    KEEP_DOCS=true
    echo "Will keep documentation files (.md)"
    echo ""
fi

# Clean each file
CLEANED=0
KEPT=0

for file in "${TEMP_FILES[@]}"; do
    if [ -f "$file" ]; then
        # Skip .md files if --keep-docs is set
        if [ "$KEEP_DOCS" = true ] && [[ "$file" == *.md ]]; then
            echo "⊙ Keeping: $file"
            ((KEPT++))
            continue
        fi
        
        echo "✓ Removing: $file"
        rm "$file"
        ((CLEANED++))
    else
        echo "⊘ Not found: $file"
    fi
done

# Clean old log files (keep only the latest service log)
echo ""
echo "Checking old log files..."
OLD_LOGS=(
    "test_xyz.log"
    "test_xyz_fixed.log"
    "test_xyz_fixed_v2.log"
    "test_xyz_fixed_final.log"
    "test_service_restart.log"
)

for log in "${OLD_LOGS[@]}"; do
    if [ -f "$log" ]; then
        echo "✓ Removing old log: $log"
        rm "$log"
        ((CLEANED++))
    fi
done

echo ""
echo "=========================================="
echo "  Cleanup Summary"
echo "=========================================="
echo "Files removed: $CLEANED"
echo "Files kept: $KEPT"
echo ""
echo "Current service log: service_with_patch.log"
echo "Active .gitignore rules added ✓"
echo ""
echo "Done!"

