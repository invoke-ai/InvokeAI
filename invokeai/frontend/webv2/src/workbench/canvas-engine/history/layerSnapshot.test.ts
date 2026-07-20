import type { CanvasControlLayerContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it, vi } from 'vitest';

import { createLayerSnapshotEntry, type LayerPixelSnapshot } from './layerSnapshot';

const fakeImageData = (width: number, height: number): ImageData =>
  ({
    colorSpace: 'srgb',
    data: new Uint8ClampedArray(width * height * 4),
    height,
    width,
  }) as unknown as ImageData;

const layer = (source: CanvasControlLayerContract['source']): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: 'm', weight: 1 },
  blendMode: 'normal',
  id: 'control',
  isEnabled: true,
  isLocked: false,
  name: 'Control',
  opacity: 1,
  source,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const snapshot = (which: 'before' | 'after'): LayerPixelSnapshot => ({
  layer: layer(
    which === 'before'
      ? { image: { height: 2, imageName: 'before', width: 3 }, type: 'image' }
      : { bitmap: null, offset: { x: 4, y: 5 }, type: 'paint' }
  ),
  pixels: fakeImageData(3, 2),
  rect: { height: 2, width: 3, x: which === 'before' ? 0 : 4, y: which === 'before' ? 0 : 5 },
});

describe('createLayerSnapshotEntry', () => {
  it('replays exact before/after snapshots and accounts for both buffers', () => {
    const before = snapshot('before');
    const after = snapshot('after');
    const apply = vi.fn();
    const entry = createLayerSnapshotEntry({ after, apply, before, label: 'Brush stroke' });

    expect(entry.bytes).toBe(before.pixels!.data.byteLength + after.pixels!.data.byteLength + 256);
    expect(entry.replayFailureAtomic).toBe(true);
    entry.undo();
    entry.redo();
    expect(apply).toHaveBeenNthCalledWith(1, before);
    expect(apply).toHaveBeenNthCalledWith(2, after);
  });

  it('rejects mismatched layer ids', () => {
    const before = snapshot('before');
    const after = { ...snapshot('after'), layer: { ...snapshot('after').layer, id: 'other' } };
    expect(() => createLayerSnapshotEntry({ after, apply: vi.fn(), before, label: 'Edit' })).toThrow(
      'layerSnapshot: before/after layer ids differ'
    );
  });

  it('rejects pixel dimensions that do not match the rect', () => {
    const before = { ...snapshot('before'), pixels: fakeImageData(1, 1) };
    expect(() => createLayerSnapshotEntry({ after: snapshot('after'), apply: vi.fn(), before, label: 'Edit' })).toThrow(
      'layerSnapshot: before pixels do not match rect'
    );
  });

  it('accepts null pixels only for an empty cache extent', () => {
    const before = { ...snapshot('before'), pixels: null, rect: { height: 0, width: 0, x: 0, y: 0 } };
    expect(() =>
      createLayerSnapshotEntry({ after: snapshot('after'), apply: vi.fn(), before, label: 'First stroke' })
    ).not.toThrow();
    expect(() =>
      createLayerSnapshotEntry({
        after: snapshot('after'),
        apply: vi.fn(),
        before: { ...snapshot('before'), pixels: null },
        label: 'Edit',
      })
    ).toThrow('layerSnapshot: before non-empty rect requires pixels');
  });

  it('protects contract and rect snapshots from later caller mutation', () => {
    const before = snapshot('before');
    const after = snapshot('after');
    const apply = vi.fn();
    const entry = createLayerSnapshotEntry({ after, apply, before, label: 'Edit' });
    before.layer.name = 'mutated';
    before.rect.x = 999;
    entry.undo();
    expect(apply).toHaveBeenCalledWith(
      expect.objectContaining({
        layer: expect.objectContaining({ name: 'Control' }),
        rect: expect.objectContaining({ x: 0 }),
      })
    );
  });
});
