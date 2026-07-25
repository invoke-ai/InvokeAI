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
    previousImageName: undefined,
    sourceChanged: false,
    ...overrides,
  });

describe('a layer that is gone', () => {
  it('is reported removed', () => {
    expect(decide({ layer: undefined }).kind).toBe('removed');
  });

  it('releases the image it referenced', () => {
    expect(decide({ layer: undefined, previousImageName: 'old.png' })).toMatchObject({
      releaseImageName: 'old.png',
    });
  });

  it('releases nothing when it referenced no image', () => {
    expect(decide({ layer: undefined })).toMatchObject({ releaseImageName: null });
  });

  it.each([
    ['transform', { hasTransformSession: true }, 'cancelTransformSession'],
    ['text-edit', { hasTextEditSession: true }, 'cancelTextEditSession'],
  ])('tears down an open %s session belonging to it', (_label, session, field) => {
    // A session outlives individual gestures, so it can outlive its own layer —
    // deleted from the layers panel, or undone — and must not be left pointing
    // at an id that no longer exists.
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
    // Invalidating here is wasteful for an image layer and destructive for an
    // unflushed paint layer, whose `bitmap: null` source rasterizes to a cleared
    // surface and would wipe strokes still only in the cache.
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
    // The cache already holds exactly those pixels; re-rasterizing would refetch
    // them and risk a flicker.
    expect(decide({ isSelfEcho: () => true, sourceChanged: true })).toMatchObject({ invalidateCache: false });
  });

  it('re-keys the thumbnail without a display token', () => {
    const layer = layerOf({ source: { image: { height: 8, imageName: 'b.png', width: 8 }, type: 'image' } });
    const decision = decide({ layer, sourceChanged: true });
    expect(decision).toMatchObject({ thumbnailKey: getLayerThumbnailDisplayKey(layer) });
  });

  it('releases the image it previously referenced, echo or not', () => {
    expect(decide({ previousImageName: 'old.png', sourceChanged: true })).toMatchObject({
      releaseImageName: 'old.png',
    });
    expect(decide({ isSelfEcho: () => true, previousImageName: 'old.png', sourceChanged: true })).toMatchObject({
      releaseImageName: 'old.png',
    });
  });

  it('re-keys the thumbnail even for an echo it will not re-rasterize', () => {
    expect(decide({ isSelfEcho: () => true, sourceChanged: true })).toMatchObject({
      thumbnailKey: expect.any(String),
    });
  });
});
