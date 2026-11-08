import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it, vi } from 'vitest';

import { createImagePatchEntry } from './imagePatch';

/** A fake ImageData carrier sized to `w`x`h` (byteLength = w*h*4). */
const fakeImageData = (w: number, h: number): ImageData =>
  ({ colorSpace: 'srgb', data: new Uint8ClampedArray(w * h * 4), height: h, width: w }) as unknown as ImageData;

const rect: Rect = { height: 3, width: 4, x: 5, y: 6 };

describe('createImagePatchEntry', () => {
  it('applies before on undo and after on redo with the patch rect', () => {
    const before = fakeImageData(4, 3);
    const after = fakeImageData(4, 3);
    const apply = vi.fn();

    const entry = createImagePatchEntry({ after, apply, before, label: 'Brush stroke', layerId: 'L1', rect });

    entry.undo();
    expect(apply).toHaveBeenNthCalledWith(1, 'L1', rect, before);
    entry.redo();
    expect(apply).toHaveBeenNthCalledWith(2, 'L1', rect, after);
  });

  it('accounts bytes as both buffers combined', () => {
    const before = fakeImageData(4, 3); // 48 bytes
    const after = fakeImageData(4, 3); // 48 bytes
    const entry = createImagePatchEntry({
      after,
      apply: vi.fn(),
      before,
      label: 'Brush stroke',
      layerId: 'L1',
      rect,
    });
    expect(entry.bytes).toBe(before.data.byteLength + after.data.byteLength);
    expect(entry.bytes).toBe(96);
    expect(entry.label).toBe('Brush stroke');
  });

  it('snapshots the rect so later caller mutation does not shift the write', () => {
    const before = fakeImageData(4, 3);
    const after = fakeImageData(4, 3);
    const apply = vi.fn();
    const mutableRect: Rect = { height: 3, width: 4, x: 5, y: 6 };
    const entry = createImagePatchEntry({
      after,
      apply,
      before,
      label: 'Brush stroke',
      layerId: 'L1',
      rect: mutableRect,
    });
    mutableRect.x = 999;
    entry.undo();
    expect(apply).toHaveBeenCalledWith('L1', { height: 3, width: 4, x: 5, y: 6 }, before);
  });

  it('throws when an ImageData dimension disagrees with the rect', () => {
    const good = fakeImageData(4, 3);
    const wrong = fakeImageData(2, 3);
    expect(() =>
      createImagePatchEntry({ after: wrong, apply: vi.fn(), before: good, label: 'x', layerId: 'L1', rect })
    ).toThrow(/does not match rect/);
    expect(() =>
      createImagePatchEntry({ after: good, apply: vi.fn(), before: wrong, label: 'x', layerId: 'L1', rect })
    ).toThrow(/before ImageData/);
  });
});
