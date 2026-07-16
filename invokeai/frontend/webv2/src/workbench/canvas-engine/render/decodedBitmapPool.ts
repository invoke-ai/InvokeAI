export interface DecodedBitmapLease {
  readonly bitmap: ImageBitmap;
  release(): void;
}

export interface DecodedBitmapPool {
  acquire(
    key: string,
    decode: (signal?: AbortSignal) => Promise<ImageBitmap>,
    signal?: AbortSignal
  ): Promise<DecodedBitmapLease>;
  byteSize(): number;
  dispose(): void;
}

interface PoolEntry {
  controller: AbortController;
  promise: Promise<ImageBitmap>;
  bitmap: ImageBitmap | null;
  bytes: number;
  leases: number;
}

const bitmapBytes = (bitmap: ImageBitmap): number => bitmap.width * bitmap.height * 4;

/** A decode-coalescing pool whose bitmaps live only while callers hold leases. */
export const createDecodedBitmapPool = (
  options: { onBytesChange?: (bytes: number) => void } = {}
): DecodedBitmapPool => {
  const entries = new Map<string, PoolEntry>();
  let totalBytes = 0;
  let disposed = false;

  const reportBytes = (): void => options.onBytesChange?.(totalBytes);

  const releaseInterest = (key: string, entry: PoolEntry): void => {
    entry.leases = Math.max(0, entry.leases - 1);
    if (entry.leases !== 0 || entries.get(key) !== entry) {
      return;
    }
    entries.delete(key);
    if (entry.bitmap) {
      entry.bitmap.close();
      totalBytes = Math.max(0, totalBytes - entry.bytes);
      entry.bitmap = null;
      entry.bytes = 0;
      reportBytes();
    } else {
      entry.controller.abort();
    }
  };

  const waitForBitmap = (
    promise: Promise<ImageBitmap>,
    signal: AbortSignal | undefined,
    releaseOnAbort: () => void
  ): Promise<ImageBitmap> => {
    if (!signal) {
      return promise;
    }
    if (signal.aborted) {
      return Promise.reject(signal.reason);
    }
    return new Promise<ImageBitmap>((resolve, reject) => {
      const onAbort = (): void => {
        signal.removeEventListener('abort', onAbort);
        releaseOnAbort();
        reject(signal.reason);
      };
      signal.addEventListener('abort', onAbort, { once: true });
      promise.then(
        (bitmap) => {
          signal.removeEventListener('abort', onAbort);
          resolve(bitmap);
        },
        (error: unknown) => {
          signal.removeEventListener('abort', onAbort);
          reject(error);
        }
      );
    });
  };

  const acquire = async (
    key: string,
    decode: (signal?: AbortSignal) => Promise<ImageBitmap>,
    signal?: AbortSignal
  ): Promise<DecodedBitmapLease> => {
    if (disposed) {
      throw new Error('DecodedBitmapPool is disposed.');
    }
    let entry = entries.get(key);
    if (!entry) {
      const created: PoolEntry = {
        bitmap: null,
        bytes: 0,
        controller: new AbortController(),
        leases: 0,
        promise: Promise.resolve(null as never),
      };
      created.promise = decode(created.controller.signal).then((bitmap) => {
        if (disposed) {
          bitmap.close();
          throw new Error('DecodedBitmapPool was disposed during decode.');
        }
        if (entries.get(key) !== created || created.leases === 0) {
          bitmap.close();
          throw created.controller.signal.reason ?? new Error('DecodedBitmapPool discarded an unowned decode.');
        }
        created.bitmap = bitmap;
        created.bytes = bitmapBytes(bitmap);
        totalBytes += created.bytes;
        reportBytes();
        return bitmap;
      });
      entry = created;
      entries.set(key, entry);
    }
    entry.leases += 1;
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      releaseInterest(key, entry);
    };
    let bitmap: ImageBitmap;
    try {
      bitmap = await waitForBitmap(entry.promise, signal, release);
    } catch (error) {
      release();
      throw error;
    }

    return { bitmap, release };
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    for (const entry of entries.values()) {
      entry.controller.abort();
      if (entry.bitmap) {
        entry.bitmap.close();
        entry.bitmap = null;
      }
    }
    entries.clear();
    if (totalBytes !== 0) {
      totalBytes = 0;
      reportBytes();
    }
  };

  return { acquire, byteSize: () => totalBytes, dispose };
};
