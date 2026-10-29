import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import { layerRowSummary } from './layerRowSummary';

const t = ((key: string, options?: { count?: number; value?: number }): string =>
  options?.count !== undefined
    ? `${key}:${options.count}`
    : options?.value !== undefined
      ? `${key}:${options.value}`
      : key) as unknown as TFunction;

describe('layerRowSummary', () => {
  it('names a raster source and only non-default blend or opacity', () => {
    expect(layerRowSummary(layerContract('r'), t)).toBe('widgets.layers.types.paint');
    expect(layerRowSummary(layerContract('r', 'raster', { blendMode: 'multiply', opacity: 0.5 }), t)).toBe(
      'widgets.layers.types.paint · widgets.layers.blendModes.multiply · 50%'
    );
  });

  it('asks a new untyped control layer for its type, a typed empty one for an image, and else names type, detector and weight', () => {
    const control = layerContract('c', 'control') as CanvasControlLayerContract;
    const image = { image: { height: 8, imageName: 'c.png', width: 8 }, type: 'image' } as const;
    expect(layerRowSummary({ ...control, source: image }, t)).toBe('widgets.layers.control.summary.noType');
    // A legacy layer that picked its model keeps the adapter summary it always had.
    expect(
      layerRowSummary({ ...control, adapter: { ...control.adapter, model: 'sdxl-depth' }, source: image }, t)
    ).toBe('widgets.layers.control.kinds.controlnet · 1.00');
    expect(layerRowSummary({ ...control, controlType: 'depth', source: { bitmap: null, type: 'paint' } }, t)).toBe(
      'widgets.layers.control.summary.noImage'
    );
    const detected = {
      ...control,
      adapter: { ...control.adapter, weight: 0.75 },
      controlType: 'depth',
      detection: { settings: {}, type: 'depth_anything_depth_estimation' },
      source: image,
    } satisfies CanvasControlLayerContract;
    expect(layerRowSummary(detected, t)).toBe(
      'widgets.layers.control.types.depth · widgets.layers.control.filters.depth_anything_depth_estimation · 0.75'
    );
    expect(layerRowSummary({ ...detected, detection: null }, t)).toBe(
      'widgets.layers.control.types.depth · common.none · 0.75'
    );
  });

  it('shows a prompt or reference count, and denoise bounds', () => {
    const regional = layerContract('g', 'regional_guidance');
    expect(layerRowSummary({ ...regional, positivePrompt: ' a red barn ' } as typeof regional, t)).toBe('a red barn');
    expect(layerRowSummary(regional, t)).toBe('widgets.layers.types.regional_guidance');
    const mask = layerContract('m', 'inpaint_mask');
    expect(
      layerRowSummary(
        { ...mask, denoise: { isEnabled: true, limit: 0.6 }, noise: { isEnabled: true, level: 0.2 } } as typeof mask,
        t
      )
    ).toBe(
      'widgets.layers.types.inpaint_mask · widgets.layers.summary.denoiseLimit:60 · widgets.layers.summary.noise:20'
    );
  });
});
