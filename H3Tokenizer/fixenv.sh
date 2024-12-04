#!/bin/bash

# Function to apply fix to a file
apply_fix() {
    local file_path=$1
    local line_number=$2
    local new_line=$3

    # Check if the file exists
    if [ ! -f "$file_path" ]; then
        echo "Error: File not found at $file_path"
        return 1
    fi

    # Create a backup of the original file
    cp "$file_path" "${file_path}.bak"
    echo "Backup created at ${file_path}.bak"

    # Apply the fix using sed
    sed -i "${line_number}c\\${new_line}" "$file_path"

    # Check if the change was applied successfully
    if grep -q "$new_line" "$file_path"; then
        echo "Fix applied successfully to $file_path"
    else
        echo "Error: Failed to apply the fix to $file_path. Please check the file manually."
        return 1
    fi

    return 0
}

# File paths
DTYPES_PATH="/opt/conda/envs/tokenizer/lib/python3.12/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py"
TENSOR_UTIL_PATH="/opt/conda/envs/tokenizer/lib/python3.12/site-packages/tensorboard/util/tensor_util.py"

# Apply fix to dtypes.py
apply_fix "$DTYPES_PATH" 677 "        if type_value.type == np.bytes_ or type_value.type == np.str_:"

# Apply fix to tensor_util.py
apply_fix "$TENSOR_UTIL_PATH" 140 "    if dtype.type == np.bytes_ or dtype.type == np.str_:"

echo "Script completed. Please verify the changes in both files."