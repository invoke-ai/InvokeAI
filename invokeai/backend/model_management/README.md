# Model Cache

## `glibc` Memory Allocator Fragmentation

Python (and PyTorch) relies on the memory allocator from the C Standard Library (`libc`). On linux, with the GNU C Standard Library implementation (`glibc`), our memory access patterns have been observed to cause severe memory fragmentation. This fragmentation results in large amounts of memory that has been freed but can't be released back to the OS. Loading models from disk and moving them between CPU/CUDA seem to be the operations that contribute most to the fragmentation. This memory fragmentation issue can result in OOM crashes during frequent model switching, even if `max_cache_size` is set to a reasonable value (e.g. a OOM crash with `max_cache_size=16` on a system with 32GB of RAM).

This problem may also exist on other OSes, and other `libc` implementations. But, at the time of writing, it has only been investigated on linux with `glibc`.

To better understand how the `glibc` memory allocator works, see these references:
- Basics: https://www.gnu.org/software/libc/manual/html_node/The-GNU-Allocator.html
- Details: https://sourceware.org/glibc/wiki/MallocInternals

Note the differences between memory allocated as chunks in an arena vs. memory allocated with `mmap`. Under `glibc`'s default configuration, most model tensors get allocated as chunks in an arena making them vulnerable to the problem of fragmentation.

We can work around this memory fragmentation issue by setting the following env var:

```bash
# Force blocks >1MB to be allocated with `mmap` so that they are released to the system immediately when they are freed.
MALLOC_MMAP_THRESHOLD_=1048576
```

See the following references for more information about the `malloc` tunable parameters:
- https://www.gnu.org/software/libc/manual/html_node/Malloc-Tunable-Parameters.html
- https://www.gnu.org/software/libc/manual/html_node/Memory-Allocation-Tunables.html
- https://man7.org/linux/man-pages/man3/mallopt.3.html

The model cache emits debug logs that provide visibility into the state of the `libc` memory allocator. See the `LibcUtil` class for more info on how these `libc` malloc stats are collected.
