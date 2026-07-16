import { describe, expect, it } from 'vitest';

import { createTestStubRasterBackend } from './raster.testStub';

describe('createTestStubRasterBackend', () => {
  it('creates a surface with the requested dimensions', () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(64, 32);
    expect(surface.width).toBe(64);
    expect(surface.height).toBe(32);
  });

  it('resize updates the surface dimensions and logs the call', () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(10, 10);
    surface.resize(20, 40);
    expect(surface.width).toBe(20);
    expect(surface.height).toBe(40);
    expect(surface.callLog).toContainEqual({ op: 'resize', args: [20, 40] });
  });

  it('records draw calls made against the surface context', () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(10, 10);
    const ctx = surface.ctx;

    ctx.save();
    ctx.clearRect(0, 0, 10, 10);
    ctx.setTransform(1, 0, 0, 1, 5, 5);
    ctx.fill();
    ctx.restore();

    const ops = surface.callLog.map((entry) => entry.op);
    expect(ops).toEqual(['save', 'clearRect', 'setTransform', 'fill', 'restore']);
    expect(surface.callLog[1]).toEqual({ op: 'clearRect', args: [0, 0, 10, 10] });
    expect(surface.callLog[2]).toEqual({ op: 'setTransform', args: [1, 0, 0, 1, 5, 5] });
  });

  it('getImageData returns a correctly-sized, zeroed buffer', () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(4, 3);
    const imageData = surface.ctx.getImageData(0, 0, 4, 3);
    expect(imageData.width).toBe(4);
    expect(imageData.height).toBe(3);
    expect(imageData.data.length).toBe(4 * 3 * 4);
    expect(Array.from(imageData.data).every((v) => v === 0)).toBe(true);
  });

  it('putImageData is recorded in the call log', () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(2, 2);
    const imageData = surface.ctx.getImageData(0, 0, 2, 2);
    surface.ctx.putImageData(imageData, 0, 0);
    expect(surface.callLog.at(-1)).toEqual({ op: 'putImageData', args: [imageData, 0, 0] });
  });

  it('createImageBitmap resolves to a bitmap-shaped object', async () => {
    const backend = createTestStubRasterBackend();
    const fakeSource = {} as ImageBitmapSource;
    const bitmap = await backend.createImageBitmap(fakeSource);
    expect(bitmap).toHaveProperty('width');
    expect(bitmap).toHaveProperty('height');
  });

  it('encodeSurface resolves to a deterministic, size-keyed image blob', async () => {
    const backend = createTestStubRasterBackend();
    const surface = backend.createSurface(8, 4);
    const blob = await backend.encodeSurface(surface);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('image/png');
    // Deterministic: same-size surfaces encode to identical bytes.
    const again = await backend.encodeSurface(backend.createSurface(8, 4));
    expect(await blob.text()).toBe(await again.text());
    // Size participates in the encoding, so different sizes differ.
    const other = await backend.encodeSurface(backend.createSurface(8, 5));
    expect(await other.text()).not.toBe(await blob.text());
  });
});
