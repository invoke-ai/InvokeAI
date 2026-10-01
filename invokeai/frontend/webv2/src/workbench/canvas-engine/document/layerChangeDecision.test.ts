import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';
import { describe, expect, it, vi } from 'vitest';

import { decideLayerChange, type LayerChangeInput } from './layerChangeDecision';

const layerOf = (overrides: Record<string, unknown> = {}): CanvasLayerContract =>
  ({
    id: 'a',
    isEnabled: true,
    name: 'a',
    opacity: 1,
    source: { image: { height: 8, imageName: 'a.png', width: 8 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...overrides,
  }) as unknown as CanvasLayerContract;

const decide = (overrides: Partial<LayerChangeInput> = {}) =>
  decideLayerChange({
    currentThumbnailKey: undefined,
    currentThumbnailVersion: undefined,
    hasTextEditSession: false,
    hasTransformSession: false,
    isSelfEcho: () => false,
    layer: layerOf(),
    sourceChanged: false,
    ...overrides,
  });

describe('a layer that is gone', () => {
  it('is reported removed', () => {
    expect(decide({ layer: undefined }).kind).toBe('removed');
  });

  it.each([
    ['transform', { hasTransformSession: true }, 'cancelTransformSession'],
    ['text-edit', { hasTextEditSession: true }, 'cancelTextEditSession'],
  ])('tears down an open %s session belonging to it', (_label, session, field) => {
    // Sessions outlive gestures; removing their layer must clear the retained id.
    expect(decide({ layer: undefined, ...session })).toMatchObject({ [field]: true });
  });

  it('leaves a session belonging to another layer alone', () => {
    expect(decide({ layer: undefined })).toMatchObject({
      cancelTextEditSession: false,
      cancelTransformSession: false,
    });
  });

  it('is removed even when its source also changed', () => {
    expect(decide({ layer: undefined, sourceChanged: true }).kind).toBe('removed');
  });
});

describe('a change that left the source alone', () => {
  it('never invalidates the cache', () => {
    // Rerasterizing a null-bitmap paint source erases unflushed cache-only strokes.
    expect(decide().kind).toBe('appearance-only');
  });

  it('records nothing when the layer looks exactly as it did', () => {
    const layer = layerOf();
    const decision = decide({ currentThumbnailKey: getLayerThumbnailDisplayKey(layer), layer });
    expect(decision).toEqual({ kind: 'appearance-only', thumbnailDisplay: null });
  });

  it('re-keys the thumbnail when the layer looks different', () => {
    const layer = layerOf({ opacity: 0.5 });
    const decision = decide({ currentThumbnailKey: 'stale-key', layer });
    expect(decision).toMatchObject({
      thumbnailDisplay: { key: getLayerThumbnailDisplayKey(layer) },
    });
  });

  it('tokens the first appearance change negative so it cannot collide with a cache version', () => {
    // Cache versions are positive; a negative token can never be mistaken for
    // one and suppress the redraw that follows the next publication.
    expect(decide({ currentThumbnailKey: 'stale-key' })).toMatchObject({
      thumbnailDisplay: { version: -1 },
    });
  });

  it('steps further negative on each subsequent appearance change', () => {
    expect(decide({ currentThumbnailKey: 'stale-key', currentThumbnailVersion: -3 })).toMatchObject({
      thumbnailDisplay: { version: -4 },
    });
  });

  it('starts a fresh negative token when the recorded version is a real cache version', () => {
    expect(decide({ currentThumbnailKey: 'stale-key', currentThumbnailVersion: 7 })).toMatchObject({
      thumbnailDisplay: { version: -1 },
    });
  });

  it('never asks whether the change was a self-echo', () => {
    const isSelfEcho = vi.fn(() => false);
    decide({ currentThumbnailKey: 'stale-key', isSelfEcho });
    expect(isSelfEcho).not.toHaveBeenCalled();
  });
});

describe('a genuine source swap', () => {
  it('invalidates the cache so the new source is rasterized', () => {
    expect(decide({ sourceChanged: true })).toMatchObject({ invalidateCache: true, kind: 'source-changed' });
  });

  it('does not invalidate the bitmap store’s own echo', () => {
    // Self-echo pixels already match the cache; rerasterization risks flicker.
    expect(decide({ isSelfEcho: () => true, sourceChanged: true })).toMatchObject({ invalidateCache: false });
  });

  it('re-keys the thumbnail without a display token', () => {
    const layer = layerOf({ source: { image: { height: 8, imageName: 'b.png', width: 8 }, type: 'image' } });
    const decision = decide({ layer, sourceChanged: true });
    expect(decision).toMatchObject({ thumbnailKey: getLayerThumbnailDisplayKey(layer) });
  });

  it('re-keys the thumbnail even for an echo it will not re-rasterize', () => {
    expect(decide({ isSelfEcho: () => true, sourceChanged: true })).toMatchObject({
      thumbnailKey: expect.any(String),
    });
  });
});
