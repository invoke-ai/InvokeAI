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

  it('shows an adapter and weight, a prompt or reference count, and denoise bounds', () => {
    expect(layerRowSummary(layerContract('c', 'control'), t)).toMatch(
      /^widgets\.layers\.control\.kinds\.\w+ · \d\.\d\d$/
    );
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
