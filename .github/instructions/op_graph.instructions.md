---
description: I explains the structure of class `OpGraph`. Provide information to understand, use its API or build algorithms upon it.
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

class `xpipe.ir.graph.OpGraph` or `xpipe.OpGraph` is a DAG which contains number of calculation nodes. Each node is a subclass of `xpipe.op.BaseOp` which provides attributes of the node.

You can consider following attributes to extract from a node:

- inputs: a list of input buffers `xpipe.memory.Buffer`.
- outputs: a list of output buffers.
