#!/usr/bin/env python3

import argparse
import gc
import io
import mmap
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch

GiB = 1024**3
MiB = 1024**2

RESULTS = []

IS_WINDOWS = os.name == "nt"

# Windows os.open() defaults to TEXT mode, which would mangle binary payloads.
O_BINARY = getattr(os, "O_BINARY", 0)
O_SEQUENTIAL = getattr(os, "O_SEQUENTIAL", 0)

try:
    import resource  # POSIX only
except ImportError:
    resource = None

try:
    import psutil
except Exception:
    psutil = None

_PROC = psutil.Process() if psutil is not None else None


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _GENERIC_READ = 0x80000000
    _FILE_SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _win_drop_file_cache(path):
    """Windows equivalent of POSIX_FADV_DONTNEED.

    Opening a file with FILE_FLAG_NO_BUFFERING makes the cache manager flush and
    purge that file's cached pages, as long as no buffered handle to it is open.
    """
    handle = _kernel32.CreateFileW(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_NO_BUFFERING,
        None,
    )
    if not handle or handle == _INVALID_HANDLE_VALUE:
        return False
    _kernel32.CloseHandle(handle)
    return True


def cache_drop_method():
    if IS_WINDOWS:
        return "FILE_FLAG_NO_BUFFERING purge"
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        return "POSIX_FADV_DONTNEED"
    return "none"


def gib(n):
    return n / GiB


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def rss_gib():
    if _PROC is not None:
        try:
            return _PROC.memory_info().rss / GiB
        except Exception:
            pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    return float("nan")


def faults():
    """(major, minor) page faults. Windows has no major/minor split, so major is None."""
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
        return r.ru_majflt, r.ru_minflt
    if _PROC is not None:
        try:
            return None, int(getattr(_PROC.memory_info(), "num_page_faults", 0))
        except Exception:
            pass
    return None, 0


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def record(name, seconds, bytes_moved=None, note="", before_faults=None):
    if bytes_moved is not None and seconds > 0:
        bw = bytes_moved / GiB / seconds
        bwstr = f"{bw:8.2f} GiB/s"
    else:
        bw = None
        bwstr = "       n/a"

    fault_text = ""
    if before_faults is not None:
        maj0, min0 = before_faults
        maj1, min1 = faults()
        if maj0 is None or maj1 is None:
            # Windows exposes a single combined page-fault counter.
            fault_text = f"   pgflt=+{min1 - min0:<25d}"
        else:
            fault_text = f"   majflt=+{maj1 - maj0:<7d} minflt=+{min1 - min0:<8d}"

    print(
        f"{name:44s} "
        f"{fmt_time(seconds):>11s}   "
        f"{bwstr}   "
        f"RSS={rss_gib():6.2f} GiB"
        f"{fault_text}" + (f"   {note}" if note else "")
    )
    RESULTS.append((name, seconds, bw, note))


def timed(name, fn, bytes_moved=None, note="", cuda_sync=False):
    if cuda_sync:
        sync_cuda()
    f0 = faults()
    t0 = time.perf_counter()
    result = fn()
    if cuda_sync:
        sync_cuda()
    dt = time.perf_counter() - t0
    record(name, dt, bytes_moved, note, f0)
    return result, dt


def describe_mount(path):
    """FSTYPE SOURCE TARGET for the filesystem holding `path`, like `findmnt -T`."""
    if not IS_WINDOWS:
        try:
            cp = subprocess.run(
                ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE,SOURCE,TARGET"],
                check=False,
                capture_output=True,
                text=True,
            )
            if cp.stdout.strip():
                return cp.stdout.strip()
        except Exception:
            pass

    if psutil is not None:
        try:
            target = str(Path(path).resolve()).lower()
            best = None
            for part in psutil.disk_partitions(all=False):
                mp = part.mountpoint.lower()
                if target.startswith(mp) and (best is None or len(mp) > len(best.mountpoint)):
                    best = part
            if best is not None:
                return f"{best.fstype} {best.device} {best.mountpoint}"
        except Exception:
            pass
    return "unknown"


def make_backing_file(tmpdir: Path, nbytes: int):
    free = shutil.disk_usage(tmpdir).free
    if free < nbytes + 2 * GiB:
        raise RuntimeError(
            f"Not enough free space in {tmpdir}: need about {gib(nbytes + 2 * GiB):.1f} GiB, have {gib(free):.1f} GiB"
        )

    fd, path_str = tempfile.mkstemp(prefix="spark_mmap_bench_", suffix=".bin", dir=tmpdir)
    path = Path(path_str)

    # One incompressible block reused across the file. Reusing it avoids spending
    # benchmark time generating tens of GiB of random data, while still avoiding
    # the special/sparse behavior that an all-zero file could trigger.
    block_size = 16 * MiB
    block = os.urandom(block_size)

    print(f"\nCreating backing file: {path}")
    print(f"File size:             {gib(nbytes):.2f} GiB")

    f0 = faults()
    t0 = time.perf_counter()
    remaining = nbytes
    try:
        while remaining:
            chunk = block if remaining >= block_size else block[:remaining]
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            remaining -= len(chunk)
        os.fsync(fd)
    finally:
        os.close(fd)

    dt = time.perf_counter() - t0
    record("create + write + fsync backing file", dt, nbytes, "actual file data written", f0)
    return path


def drop_file_cache(path: Path):
    """Best-effort eviction of this file's clean pages from the OS page cache."""
    if IS_WINDOWS:
        return _win_drop_file_cache(path)
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    fd = os.open(path, os.O_RDONLY | O_BINARY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def advise_sequential(fd):
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)


def map_private(path: Path, sequential=False):
    # ACCESS_COPY gives us a writable Python buffer backed by MAP_PRIVATE, which
    # lets torch.frombuffer() use it without a read-only-buffer warning. We never
    # modify it, so no COW pages should be created.
    fd = os.open(path, os.O_RDONLY | O_BINARY | (O_SEQUENTIAL if sequential else 0))
    if sequential:
        advise_sequential(fd)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    if sequential and hasattr(mm, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
        mm.madvise(mmap.MADV_SEQUENTIAL)
    return fd, mm


def close_mapping(fd, mm, *objs):
    # torch.frombuffer retains the mapping. Destroy Tensor views before mmap.close().
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        mm.close()
    finally:
        os.close(fd)


def raw_read_test(path: Path, nbytes: int):
    print("\n=== File I/O baseline: read() into reusable buffer ===")
    buf = bytearray(64 * MiB)
    mv = memoryview(buf)

    def read_all():
        fd = os.open(path, os.O_RDONLY | O_BINARY | O_SEQUENTIAL)
        total = 0
        try:
            advise_sequential(fd)
            # readinto() avoids a fresh allocation per chunk and, unlike os.readv(),
            # exists on Windows. Both issue a single read() per call.
            f = io.FileIO(fd, closefd=False)
            while total < nbytes:
                want = min(len(mv), nbytes - total)
                n = f.readinto(mv[:want])
                if not n:
                    break
                total += n
        finally:
            os.close(fd)
        return total

    drop_file_cache(path)
    total, _ = timed("cold file read", read_all, nbytes, "SSD + page cache + memcpy")
    if total != nbytes:
        raise RuntimeError(f"short read: {total} / {nbytes}")

    total, _ = timed("warm file read", read_all, nbytes, "should mostly hit page cache")
    if total != nbytes:
        raise RuntimeError(f"short read: {total} / {nbytes}")


def mmap_map_only_test(path: Path, nbytes: int):
    print("\n=== mmap setup: mapping is lazy ===")
    drop_file_cache(path)
    f0 = faults()
    t0 = time.perf_counter()
    fd, mm = map_private(path)
    src = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)
    dt = time.perf_counter() - t0
    record("mmap + torch.frombuffer()", dt, None, "should not read the file", f0)
    del src
    gc.collect()
    mm.close()
    os.close(fd)


def mmap_cpu_copy_test(path: Path, nbytes: int):
    print("\n=== mmap -> preallocated CPU tensor ===")
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cpu")

    drop_file_cache(path)
    fd, mm = map_private(path, sequential=True)
    src = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)

    timed(
        "cold mmap -> CPU copy_()",
        lambda: dst.copy_(src),
        nbytes,
        "faults file pages + writes CPU destination",
    )
    timed(
        "warm mmap -> CPU copy_()",
        lambda: dst.copy_(src),
        nbytes,
        "source should now be page-cached",
    )

    del src, dst
    gc.collect()
    mm.close()
    os.close(fd)


def mmap_to_cuda_alloc_test(path: Path, nbytes: int):
    print("\n=== mmap -> CUDA via .to(): allocation + transfer ===")

    drop_file_cache(path)
    fd, mm = map_private(path, sequential=True)
    src = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)

    dst, _ = timed(
        "cold mmap -> CUDA .to()",
        lambda: src.to("cuda"),
        nbytes,
        "file faults + CUDA allocation + copy",
        cuda_sync=True,
    )
    del dst
    cleanup_cuda()

    # Do NOT evict file pages: this is the warm-page-cache comparison.
    dst, _ = timed(
        "warm mmap -> CUDA .to()",
        lambda: src.to("cuda"),
        nbytes,
        "CUDA allocation + copy; file should be cached",
        cuda_sync=True,
    )

    del dst, src
    cleanup_cuda()
    mm.close()
    os.close(fd)


def mmap_to_cuda_prealloc_test(path: Path, nbytes: int):
    print("\n=== mmap -> preallocated CUDA: isolates file/page-fault cost ===")
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    sync_cuda()

    drop_file_cache(path)
    fd, mm = map_private(path, sequential=True)
    src = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)

    timed(
        "cold mmap -> prealloc CUDA copy_()",
        lambda: dst.copy_(src),
        nbytes,
        "SSD/page faults + copy; no CUDA allocation in timed region",
        cuda_sync=True,
    )
    timed(
        "warm mmap -> prealloc CUDA copy_()",
        lambda: dst.copy_(src),
        nbytes,
        "page-cache + copy only",
        cuda_sync=True,
    )

    del src, dst
    cleanup_cuda()
    mm.close()
    os.close(fd)


def mmap_dtype_conversion_test(path: Path, nbytes: int):
    print("\n=== mmap float32 -> CUDA float16 ===")
    usable = nbytes - (nbytes % 4)
    count = usable // 4
    dst_bytes = count * 2

    print(
        f"mmap fp32 source: {gib(usable):.2f} GiB\n"
        f"CUDA fp16 dest:   {gib(dst_bytes):.2f} GiB\n"
        f"combined memory:  ~{gib(usable + dst_bytes):.2f} GiB when source pages are resident"
    )

    drop_file_cache(path)
    fd, mm = map_private(path, sequential=True)
    src = torch.frombuffer(mm, dtype=torch.float32, count=count)
    dst = torch.empty(count, dtype=torch.float16, device="cuda")
    sync_cuda()

    # Effective traffic metric = source bytes consumed + destination bytes produced.
    traffic = usable + dst_bytes

    timed(
        "cold mmap fp32 -> CUDA fp16",
        lambda: dst.copy_(src),
        traffic,
        "file faults + transfer + dtype conversion",
        cuda_sync=True,
    )
    timed(
        "warm mmap fp32 -> CUDA fp16",
        lambda: dst.copy_(src),
        traffic,
        "page-cached source + dtype conversion",
        cuda_sync=True,
    )

    del src, dst
    cleanup_cuda()
    mm.close()
    os.close(fd)


def mmap_many_tensor_test(path: Path, nbytes: int, tensor_count: int):
    print(f"\n=== mmap -> CUDA, {tensor_count:,} separate tensor views ===")
    base = nbytes // tensor_count
    # Keep all but last chunk exact; uint8 means no alignment restriction.

    drop_file_cache(path)
    fd, mm = map_private(path, sequential=True)

    src = []
    offset = 0
    for i in range(tensor_count):
        length = base if i < tensor_count - 1 else nbytes - offset
        src.append(torch.frombuffer(mm, dtype=torch.uint8, count=length, offset=offset))
        offset += length

    sync_cuda()
    f0 = faults()
    t0 = time.perf_counter()
    dst = [x.to("cuda") for x in src]
    sync_cuda()
    dt = time.perf_counter() - t0
    record(
        f"cold mmap {tensor_count:,} x .to(cuda)",
        dt,
        nbytes,
        f"avg={nbytes / tensor_count / MiB:.3f} MiB/tensor",
        f0,
    )

    del dst
    cleanup_cuda()

    sync_cuda()
    f0 = faults()
    t0 = time.perf_counter()
    dst = [x.to("cuda") for x in src]
    sync_cuda()
    dt = time.perf_counter() - t0
    record(
        f"warm mmap {tensor_count:,} x .to(cuda)",
        dt,
        nbytes,
        f"avg={nbytes / tensor_count / MiB:.3f} MiB/tensor",
        f0,
    )

    del dst, src
    cleanup_cuda()
    mm.close()
    os.close(fd)


def torch_from_file_test(path: Path, nbytes: int):
    print("\n=== torch.from_file() mmap storage -> CUDA ===")
    print("This uses PyTorch's own file-backed Storage mmap path.")

    drop_file_cache(path)
    f0 = faults()
    t0 = time.perf_counter()
    try:
        src = torch.from_file(str(path), shared=False, size=nbytes, dtype=torch.uint8)
    except Exception as e:
        print("SKIPPED torch.from_file(): unsupported on this platform/build:")
        print(" ", repr(e))
        return
    dt = time.perf_counter() - t0
    record("torch.from_file() map only", dt, None, "lazy mapping; should not read payload", f0)

    dst, _ = timed(
        "cold torch.from_file() -> CUDA",
        lambda: src.to("cuda"),
        nbytes,
        "PyTorch mmap storage + file faults + CUDA allocation/copy",
        cuda_sync=True,
    )
    del dst
    cleanup_cuda()

    dst, _ = timed(
        "warm torch.from_file() -> CUDA",
        lambda: src.to("cuda"),
        nbytes,
        "same mapped Storage with warm file pages",
        cuda_sync=True,
    )

    del dst, src
    cleanup_cuda()


def print_system_info(tmpdir: Path):
    print("=" * 92)
    print("DGX Spark / PyTorch mmap + storage benchmark")
    print("=" * 92)
    print(f"Python:              {platform.python_version()}")
    print(f"PyTorch:             {torch.__version__}")
    print(f"CUDA runtime:        {torch.version.cuda}")
    print(f"CUDA available:      {torch.cuda.is_available()}")
    print(f"GPU:                 {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}")
    print(f"Temporary directory: {tmpdir}")
    print(f"Filesystem:          {describe_mount(tmpdir)}")
    free = shutil.disk_usage(tmpdir).free
    print(f"Free filesystem:     {gib(free):.2f} GiB")
    print(f"OS:                  {platform.system()} {platform.release()}")
    print(f"Cache eviction:      {cache_drop_method()}")
    print("=" * 92)


def print_summary():
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for name, seconds, bw, note in RESULTS:
        bw_text = "-" if bw is None else f"{bw:.2f} GiB/s"
        print(f"{name:48s} {seconds:9.3f} s   {bw_text:>12s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--working-gib",
        type=float,
        default=20.0,
        help=(
            "Approximate memory working-set target. The mmap file is half this size so "
            "file-backed source pages + same-sized destination are about this value. Default: 20."
        ),
    )
    p.add_argument(
        "--tmpdir",
        default=tempfile.gettempdir() if IS_WINDOWS else "/tmp",
        help="Directory for temporary backing file",
    )
    p.add_argument("--many", type=int, default=4096, help="Tensor count for model-like mmap test")
    p.add_argument("--keep-file", action="store_true", help="Do not unlink the temporary file")
    p.add_argument("--quick", action="store_true", help="Skip dtype and many-tensor tests")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    tmpdir = Path(args.tmpdir).resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    print_system_info(tmpdir)

    fs_info = describe_mount(tmpdir).lower()
    if "tmpfs" in fs_info or "ramfs" in fs_info:
        print("\nWARNING: /tmp is RAM-backed on this system. Cold-file numbers will NOT measure SSD speed.\n")

    # A 10 GiB mapping + a 10 GiB CPU/CUDA destination is about a 20 GiB working set.
    file_bytes = int(args.working_gib * GiB / 2)
    file_bytes -= file_bytes % 4096

    print(
        f"\nRequested working set: ~{args.working_gib:.2f} GiB\n"
        f"Backing file/payload:  {gib(file_bytes):.2f} GiB\n"
        f"Typical pair:          {gib(file_bytes):.2f} GiB mmap/page-cache + "
        f"{gib(file_bytes):.2f} GiB destination\n"
    )

    path = None
    try:
        path = make_backing_file(tmpdir, file_bytes)

        if drop_file_cache(path):
            print(f"Best-effort per-file page-cache eviction: {cache_drop_method()} enabled")
        else:
            print(f"WARNING: {cache_drop_method()} unavailable; 'cold' tests may remain cached")

        raw_read_test(path, file_bytes)
        mmap_map_only_test(path, file_bytes)
        mmap_cpu_copy_test(path, file_bytes)
        mmap_to_cuda_alloc_test(path, file_bytes)
        mmap_to_cuda_prealloc_test(path, file_bytes)
        torch_from_file_test(path, file_bytes)

        if not args.quick:
            mmap_dtype_conversion_test(path, file_bytes)
            mmap_many_tensor_test(path, file_bytes, args.many)

        print_summary()

        print("\nInterpretation hints:")
        print("  * Cold near SSD speed + warm tens of GiB/s => storage/page faults dominate cold load.")
        print("  * Cold and warm both slow => CUDA allocation/copy/conversion path is the issue.")
        print("  * Warm preallocated CUDA should approach your earlier ~55 GiB/s pageable-H2D result.")
        print("  * Large +majflt on cold tests and near-zero +majflt on warm tests confirms real disk faults.")
        print("  * If cold read is far above your SSD ceiling with ~0 major faults, cache eviction did not stick.")

    finally:
        if path is not None and path.exists():
            if args.keep_file:
                print(f"Keeping temporary file: {path}")
            else:
                try:
                    path.unlink()
                    print(f"Removed temporary file: {path}")
                except Exception as e:
                    print(f"WARNING: could not remove {path}: {e}")


if __name__ == "__main__":
    main()
