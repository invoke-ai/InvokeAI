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
});
