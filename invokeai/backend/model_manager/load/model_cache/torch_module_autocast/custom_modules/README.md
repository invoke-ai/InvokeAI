
This directory contains custom implementations of common torch.nn.Module classes that add support for:
- Streaming weights to the execution device
- Applying sidecar patches at execution time (e.g. sidecar LoRA layers)

Each custom class sub-classes the original module type that is is replacing, so the following properties are preserved:
- `isinstance(m, torch.nn.OrginalModule)` should still work.
- Patching the weights directly (e.g. for LoRA) should still work. (Of course, this is not possible for quantized layers, hence the sidecar support.)
