import { describe, expect, it, vi } from 'vitest';

import { createDecodedBitmapPool } from './decodedBitmapPool';

const bitmap = () => {
  const close = vi.fn();
  return { close, height: 20, width: 10 } as unknown as ImageBitmap;
};

describe('DecodedBitmapPool', () => {
  it('coalesces concurrent decodes and closes the bitmap after the final lease', async () => {
    const decoded = bitmap();
    const decode = vi.fn(() => Promise.resolve(decoded));
    const byteChanges: number[] = [];
    const pool = createDecodedBitmapPool({ onBytesChange: (bytes) => byteChanges.push(bytes) });

    const [first, second] = await Promise.all([pool.acquire('image', decode), pool.acquire('image', decode)]);

    expect(decode).toHaveBeenCalledTimes(1);
    expect(pool.byteSize()).toBe(800);
    first.release();
    first.release();
    expect(decoded.close).not.toHaveBeenCalled();
    second.release();
    second.release();
    expect(decoded.close).toHaveBeenCalledTimes(1);
    expect(pool.byteSize()).toBe(0);
    expect(byteChanges).toEqual([800, 0]);
  });

  it('closes a decode that completes after disposal', async () => {
    let resolve!: (value: ImageBitmap) => void;
    const decode = () =>
      new Promise<ImageBitmap>((next) => {
        resolve = next;
      });
    const pool = createDecodedBitmapPool();
    const pending = pool.acquire('late', decode);
    pool.dispose();
    const decoded = bitmap();
    resolve(decoded);

    await expect(pending).rejects.toThrow(/disposed/i);
    expect(decoded.close).toHaveBeenCalledTimes(1);
  });

  it('refuses an already-aborted request before starting any decode', async () => {
    const decode = vi.fn(() => Promise.resolve(bitmap()));
    const pool = createDecodedBitmapPool();
    const controller = new AbortController();
    controller.abort(new Error('already cancelled'));

    await expect(pool.acquire('cancelled', decode, controller.signal)).rejects.toThrow('already cancelled');
    expect(decode).not.toHaveBeenCalled();
    expect(pool.byteSize()).toBe(0);
  });

  it('keeps a shared decode for the remaining waiter when another cancels', async () => {
    const decoded = bitmap();
    let resolve!: (value: ImageBitmap) => void;
    const decode = vi.fn(
      () =>
        new Promise<ImageBitmap>((next) => {
          resolve = next;
        })
    );
    const pool = createDecodedBitmapPool();
    const cancelled = new AbortController();
    const first = pool.acquire('shared', decode, cancelled.signal);
    const second = pool.acquire('shared', decode);
    cancelled.abort(new Error('first left'));
    await expect(first).rejects.toThrow('first left');

    resolve(decoded);
    const lease = await second;
    expect(lease.bitmap).toBe(decoded);
    expect(decode).toHaveBeenCalledOnce();
    lease.release();
    expect(decoded.close).toHaveBeenCalledOnce();
  });
});
