---
description: I explains the structure of Xe dumped graph (XeGraph), which is a description file in JSON format.
applyTo: Implement functions/methods related to "XE" or "Xe Graph".
---

## Basic Structure
```json
{
  "inputs": [],
  "outputs": [],
  "constants": [],
  "variants": [],
  "kernels": [],
  "bin": ""
}
```

### 1. Inputs/Outputs
Inputs(Outputs) contain a list of JSON object to represent IO buffers, each follows the format:
```json
{
  "id": 0,
  "name": "name",
  "kernels": ["a", "b"],
  "arg_index": [0, 1],
  "size": 1,
  "shape": [1],
  "layout": "bfyx",
  "dtype": "f16"
}
```
"id" is a unique integer representing the buffer. "name" is a human readable tag. "kernels" are a list of name for which kernel (from "kernels") uses this input buffer. "arg_index" are a list of integers for argument index of the kernel uses the input buffer, the order should follow the "kernels" list. "size" is the bytes in total. "shape" is the tensor shape. "layout" is a string for the memory layout. "dtype" is the data type.

Outputs is almost the same with inputs, except there is no "name" in outputs.

### 2. Constants
Constants represent weights buffer, each follows the format:
```json
{
  "id": 18446664917540667392,
  "size": 18816,
  "offset": 0,
  "type": "global"
}
```
"id" is a unique integer for the buffer. "size" is the total bytes. "offset" is the offset position in "bin" binary. "type" is the runtime tag for this constant buffer.

### 3. Variants
Variants represent activation buffer:
```json
{
  "id": 18446664917594865664,
  "size": 1600000
}
```
"id" is a unique integer for the buffer. "size" is the total bytes.

### 4. Kernels
Kernels represent Xe (GPU) kernel details.
```json
{
  "input": false,
  "output": false,
  "input_name": "",
  "id": "x",
  "unique_id": "x_0",
  "entry": "func",
  "domain": null,
  "group_size": [],
  "local_size": [],
  "arguments": [],
  "depends": ["a", "b"],
  "barrier": false,
  "output_layouts": ["bfyx"]
}
```
"input" ("output") is a flag to show if this kernel is input (output). "input_name" show which input this kernel uses. "id" is the name from original network. "unique_id" is a unique name for kernel. "entry" is the name of entry function. "domain" is an optional string to identify entry functions. "group_size" and "local_size" are launch parameters. "arguments" are a list of argument object. "depends" are a list of dependent kernel names. "barrier" is a flag to show whether to insert barrier after this kernel. "output_layouts" is a debug info.

#### Argument
Argument represent details for kernel argument.
```json
{
  "mem_id": 1886844485632,
  "type": "INPUT",
  "dtype": "f16",
  "shape": [1,2,3,4],
  "is_input": true,
  "layout": "bfyx",
  "block": {
    "block_sizes": [],
    "block_idx": [],
    "strides": []
  },
  "memory_type": "gpu_usm",
  "allocation_type": "usm_host",
  "alias_offset": 0
}
```
"mem_id" is the unique integer for this buffer. "type" is the usage of this argument. "dtype" is the data type. "shape" is tensor shape. "is_input" is a flag to show if this argument is input buffer. "layout" and "block" show how tensor is placed in the memory. "memory_type" and "allocation_type" are debug info. "alias_offset" is the offset for the buffer pointer, the access of this buffer may not start at 0.

### 5. Bin
Bin is a base64 encoded binary data for weights.
