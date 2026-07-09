import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it } from 'vitest';

import { getLayerContextActions } from './layerContextActions';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
} from './layerOps';

const paintLayer = (id: string, patch: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  ...createEmptyPaintLayer(id, id),
  ...patch,
});

const idsFor = (layer: CanvasLayerContract, layers: readonly CanvasLayerContract[] = [layer], hasEngine = true) =>
  getLayerContextActions({ hasEngine, index: layers.indexOf(layer), layer, layers }).map((action) => action.id);

describe('getLayerContextActions', () => {
  it('keeps common actions available for a raster layer', () => {
    expect(idsFor(paintLayer('raster'))).toEqual(
      expect.arrayContaining([
        'move-to-front',
        'duplicate',
        'rename',
        'transform',
        'fit-to-bbox',
        'save-to-assets',
        'copy-to-clipboard',
        'crop-to-bbox',
        'merge-down',
        'delete',
      ])
    );
  });

  it('exposes only pixel-backed raster to control conversion', () => {
    const paint = paintLayer('paint');
    const text = paintLayer('text', {
      source: {
        align: 'left',
        color: '#fff',
        content: 'hello',
        fontFamily: 'Inter',
        fontSize: 32,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      },
    });

    expect(idsFor(paint)).toContain('convert-to-control');
    expect(idsFor(text)).not.toContain('convert-to-control');
  });

  it('exposes control-only transparency and raster conversion actions', () => {
    const layer = createControlLayer('Control', 'control');

    expect(idsFor(layer)).toEqual(
      expect.arrayContaining(['control-transparency-effect', 'convert-to-raster', 'copy-to-raster'])
    );
  });

  it('does not expose copy-to-raster for raster layers', () => {
    expect(idsFor(paintLayer('raster'))).not.toContain('copy-to-raster');
  });

  it('exposes regional guidance add actions only for missing prompts', () => {
    const layer = { ...createRegionalGuidanceLayer('Region', 0, 'region'), positivePrompt: 'already' };

    expect(idsFor(layer)).toContain('regional-negative-prompt');
    expect(idsFor(layer)).toContain('regional-reference-image');
    expect(idsFor(layer)).toContain('regional-auto-negative');
    expect(idsFor(layer)).not.toContain('regional-positive-prompt');
  });

  it('exposes inpaint modifier add actions only for missing modifiers', () => {
    const layer = { ...createInpaintMaskLayer('Mask', 'mask'), noiseLevel: 0.25 };

    expect(idsFor(layer)).toContain('inpaint-denoise-limit');
    expect(idsFor(layer)).not.toContain('inpaint-noise');
  });

  it('disables transform and merge-down without an engine', () => {
    const layer = paintLayer('raster');
    const actions = getLayerContextActions({ hasEngine: false, index: 0, layer, layers: [layer] });

    expect(actions.find((action) => action.id === 'transform')?.isDisabled).toBe(true);
    expect(actions.find((action) => action.id === 'save-to-assets')?.isDisabled).toBe(true);
    expect(actions.find((action) => action.id === 'copy-to-clipboard')?.isDisabled).toBe(true);
    expect(actions.find((action) => action.id === 'crop-to-bbox')?.isDisabled).toBe(true);
    expect(actions.find((action) => action.id === 'merge-down')?.isDisabled).toBe(true);

    const control = createControlLayer('Control', 'control');
    const controlActions = getLayerContextActions({ hasEngine: false, index: 0, layer: control, layers: [control] });
    expect(controlActions.find((action) => action.id === 'copy-to-raster')?.isDisabled).toBe(true);
  });

  it('disables crop-to-bbox for locked layers', () => {
    const layer = paintLayer('raster', { isLocked: true });
    const actions = getLayerContextActions({ hasEngine: true, index: 0, layer, layers: [layer] });

    expect(actions.find((action) => action.id === 'crop-to-bbox')?.isDisabled).toBe(true);
  });
});
