import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it, vi } from 'vitest';

import { createSelectObjectBridge, type CreateSelectObjectBridgeDeps } from './selectObjectBridge';

const layer = (overrides: Record<string, unknown> = {}) =>
  ({
    id: 'layer-1',
    isEnabled: true,
    isLocked: false,
    name: 'Layer 1',
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 },
    type: 'raster',
    ...overrides,
  }) as never;

const documentOf = (layers: unknown[]): CanvasDocumentContractV2 => ({ layers }) as CanvasDocumentContractV2;

const surface = () => {
  const ctx = { fillRect: vi.fn(), fillStyle: '', globalCompositeOperation: '' };
  return { ctx, height: 8, width: 8 } as never;
};

const makeDeps = (overrides: Partial<CreateSelectObjectBridgeDeps> = {}): CreateSelectObjectBridgeDeps => ({
  captureGuard: vi.fn(() => ({ layerId: 'layer-1' }) as never),
  decodeImage: vi.fn(() => Promise.resolve({ status: 'ok', surface: surface() } as never)),
  getDocument: () => documentOf([layer()]),
  layerCache: { get: vi.fn(() => ({ rect: { height: 4, width: 4, x: 0, y: 0 } })) } as never,
  ...overrides,
});

const samResult = (image: Partial<CanvasImageRef> = {}, rect: Partial<Rect> = {}) => ({
  image: { height: 8, imageName: 'sam.png', width: 8, ...image } as CanvasImageRef,
  rect: { height: 8, width: 8, x: 0, y: 0, ...rect } as Rect,
});

describe('prepareSelectObjectStart', () => {
  it('reports ready with the layer identity and its document-space bounds', () => {
    const result = createSelectObjectBridge(makeDeps()).prepareSelectObjectStart('layer-1');
    expect(result).toMatchObject({
      layerId: 'layer-1',
      layerName: 'Layer 1',
      layerType: 'raster',
      sourceRect: { height: 4, width: 4, x: 10, y: 20 },
      status: 'ready',
    });
  });

  it.each([
    ['missing', { getDocument: () => documentOf([]) }],
    ['unsupported', { getDocument: () => documentOf([layer({ type: 'inpaint_mask' })]) }],
    ['disabled', { getDocument: () => documentOf([layer({ isEnabled: false })]) }],
    ['locked', { getDocument: () => documentOf([layer({ isLocked: true })]) }],
    ['not-ready', { captureGuard: vi.fn(() => null) }],
    ['not-ready', { layerCache: { get: vi.fn(() => undefined) } as never }],
  ])('refuses with %s', (status, overrides) => {
    const result = createSelectObjectBridge(makeDeps(overrides)).prepareSelectObjectStart('layer-1');
    expect(result.status).toBe(status);
  });

  it('refuses a layer whose cached content transforms to an empty rect', () => {
    const deps = makeDeps({
      layerCache: { get: vi.fn(() => ({ rect: { height: 0, width: 0, x: 0, y: 0 } })) } as never,
    });
    expect(createSelectObjectBridge(deps).prepareSelectObjectStart('layer-1').status).toBe('not-ready');
  });

  it('accepts control layers as well as rasters', () => {
    const deps = makeDeps({ getDocument: () => documentOf([layer({ type: 'control' })]) });
    expect(createSelectObjectBridge(deps).prepareSelectObjectStart('layer-1')).toMatchObject({
      layerType: 'control',
      status: 'ready',
    });
  });
});

describe('decodeSelectObjectPreview', () => {
  const signal = () => new AbortController().signal;

  it('tints the decoded mask through source-in and restores the composite mode', async () => {
    const deps = makeDeps();
    const result = await createSelectObjectBridge(deps).decodeSelectObjectPreview(samResult(), signal());
    expect(result.ctx.fillRect).toHaveBeenCalledWith(0, 0, 8, 8);
    expect(result.ctx.globalCompositeOperation).toBe('source-over');
  });

  it('rejects a decode whose dimensions disagree with the SAM output', async () => {
    const deps = makeDeps({
      decodeImage: vi.fn((_image, options) => {
        options.validateDecoded?.(4, 4);
        return Promise.resolve({ status: 'ok', surface: surface() } as never);
      }),
    });
    await expect(createSelectObjectBridge(deps).decodeSelectObjectPreview(samResult(), signal())).rejects.toMatchObject(
      { samErrorCode: 'output-dimension' }
    );
  });

  it('rejects a decode whose dimensions disagree with the preview rect', async () => {
    const deps = makeDeps({
      decodeImage: vi.fn((_image, options) => {
        options.validateDecoded?.(8, 8);
        return Promise.resolve({ status: 'ok', surface: surface() } as never);
      }),
    });
    await expect(
      createSelectObjectBridge(deps).decodeSelectObjectPreview(samResult({}, { width: 16 }), signal())
    ).rejects.toMatchObject({ samErrorCode: 'output-dimension' });
  });

  it('aborts when the decode does not complete', async () => {
    const deps = makeDeps({ decodeImage: vi.fn(() => Promise.resolve({ status: 'aborted' } as never)) });
    await expect(createSelectObjectBridge(deps).decodeSelectObjectPreview(samResult(), signal())).rejects.toThrow(
      /aborted/i
    );
  });

  it('aborts when the signal fires after a successful decode', async () => {
    const controller = new AbortController();
    const deps = makeDeps({
      decodeImage: vi.fn(() => {
        controller.abort();
        return Promise.resolve({ status: 'ok', surface: surface() } as never);
      }),
    });
    await expect(
      createSelectObjectBridge(deps).decodeSelectObjectPreview(samResult(), controller.signal)
    ).rejects.toThrow(/aborted/i);
  });
});
