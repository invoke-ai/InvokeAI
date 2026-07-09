import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { describe, expect, it, vi } from 'vitest';

import {
  applyStructural,
  canConvertRasterControl,
  canMergeLayerDown,
  convertRasterControlLayer,
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
  createRegionalGuidanceLayerWithRefImage,
  createRegionalReferenceImage,
  DEFAULT_CONTROL_ADAPTER,
  DEFAULT_INPAINT_MASK_FILL,
  fitLayerTransformToBbox,
  getControlTransparencyEffectPatch,
  getInpaintDenoiseLimitPatch,
  getInpaintNoisePatch,
  getRegionalGuidanceAutoNegativePatch,
  getRegionalGuidanceNegativePromptPatch,
  getRegionalGuidancePositivePromptPatch,
  getRegionalGuidanceReferenceImagePatch,
  isMergeableRasterLayer,
  nextControlLayerName,
  nextInpaintMaskName,
  nextRegionalGuidanceFillColor,
  REGIONAL_GUIDANCE_FILL_COLORS,
} from './layerOps';

const paintLayer = (id: string, patch: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  ...createEmptyPaintLayer(id, id),
  ...patch,
});

const imageLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const maskLayer = (id: string): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#f00', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

describe('createEmptyPaintLayer', () => {
  it('builds an enabled, document-sized paint raster layer', () => {
    const layer = createEmptyPaintLayer('Layer 1', 'l1');
    expect(layer).toMatchObject({
      id: 'l1',
      isEnabled: true,
      isLocked: false,
      name: 'Layer 1',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      type: 'raster',
    });
  });

  it('mints a unique id when none is supplied', () => {
    expect(createEmptyPaintLayer('a').id).not.toBe(createEmptyPaintLayer('a').id);
  });
});

describe('nextRegionalGuidanceFillColor', () => {
  it('derives the colour purely from the existing regional-guidance count (no session state)', () => {
    // Legacy pre-increment cycler: the first region (0 existing) gets index 1.
    expect(nextRegionalGuidanceFillColor(0)).toBe(REGIONAL_GUIDANCE_FILL_COLORS[1]);
    expect(nextRegionalGuidanceFillColor(1)).toBe(REGIONAL_GUIDANCE_FILL_COLORS[2]);
    // Wraps modulo the palette length.
    expect(nextRegionalGuidanceFillColor(REGIONAL_GUIDANCE_FILL_COLORS.length - 1)).toBe(
      REGIONAL_GUIDANCE_FILL_COLORS[0]
    );
    // Pure: the same count always yields the same colour, regardless of call order.
    expect(nextRegionalGuidanceFillColor(0)).toBe(nextRegionalGuidanceFillColor(0));
  });

  it('createRegionalGuidanceLayer uses the count-derived fill colour', () => {
    expect(createRegionalGuidanceLayer('Regional Guidance 1', 0).mask.fill.color).toBe(
      REGIONAL_GUIDANCE_FILL_COLORS[1]
    );
    expect(createRegionalGuidanceLayer('Regional Guidance 2', 1).mask.fill.color).toBe(
      REGIONAL_GUIDANCE_FILL_COLORS[2]
    );
  });
});

describe('isMergeableRasterLayer', () => {
  it('accepts enabled paint and image raster layers', () => {
    expect(isMergeableRasterLayer(paintLayer('p'))).toBe(true);
    expect(isMergeableRasterLayer(imageLayer('i'))).toBe(true);
  });

  it('rejects disabled layers and non-raster layers', () => {
    expect(isMergeableRasterLayer(paintLayer('p', { isEnabled: false }))).toBe(false);
    expect(isMergeableRasterLayer(maskLayer('m'))).toBe(false);
  });

  it('rejects locked layers (matches the paint tool: a locked target refuses edits)', () => {
    expect(isMergeableRasterLayer(paintLayer('p', { isLocked: true }))).toBe(false);
    expect(isMergeableRasterLayer({ ...imageLayer('i'), isLocked: true })).toBe(false);
  });
});

describe('canMergeLayerDown', () => {
  const layers = [paintLayer('top'), imageLayer('mid'), maskLayer('bottom')];

  it('requires an engine (merge is pixel work)', () => {
    expect(canMergeLayerDown(layers, 0, false)).toBe(false);
  });

  it('allows merging when the layer and the one below are both mergeable', () => {
    expect(canMergeLayerDown(layers, 0, true)).toBe(true);
  });

  it('disallows merging into a non-mergeable below layer', () => {
    // index 1 (mid, image) sits above the mask layer, which cannot be merged into.
    expect(canMergeLayerDown(layers, 1, true)).toBe(false);
  });

  it('disallows merging the bottom-most layer (nothing below)', () => {
    expect(canMergeLayerDown(layers, 2, true)).toBe(false);
  });

  it('disallows merging when the upper layer is locked', () => {
    const lockedTop = [paintLayer('top', { isLocked: true }), imageLayer('mid'), maskLayer('bottom')];
    expect(canMergeLayerDown(lockedTop, 0, true)).toBe(false);
  });

  it('disallows merging when the below layer is locked', () => {
    const lockedMid = [paintLayer('top'), { ...imageLayer('mid'), isLocked: true }, maskLayer('bottom')];
    expect(canMergeLayerDown(lockedMid, 0, true)).toBe(false);
  });
});

describe('applyStructural', () => {
  const forward: WorkbenchAction = { ids: ['a'], type: 'removeCanvasLayers' };
  const inverse: WorkbenchAction = { index: 0, layer: paintLayer('a'), type: 'addCanvasLayer' };

  it('routes through the engine history when an engine is attached', () => {
    const commitStructural = vi.fn();
    const engine = { commitStructural } as unknown as CanvasEngine;
    const dispatch = vi.fn();

    applyStructural(engine, dispatch, 'Delete layer', forward, inverse);

    expect(commitStructural).toHaveBeenCalledWith('Delete layer', forward, inverse);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('falls back to a plain forward dispatch without an engine', () => {
    const dispatch = vi.fn();

    applyStructural(null, dispatch, 'Delete layer', forward, inverse);

    expect(dispatch).toHaveBeenCalledWith(forward);
    expect(dispatch).toHaveBeenCalledTimes(1);
  });
});

describe('createInpaintMaskLayer', () => {
  it('builds an empty inpaint mask with the legacy-default diagonal fill', () => {
    const layer = createInpaintMaskLayer('Inpaint Mask 1', 'm1');
    expect(layer.type).toBe('inpaint_mask');
    expect(layer.id).toBe('m1');
    expect(layer.name).toBe('Inpaint Mask 1');
    expect(layer.isEnabled).toBe(true);
    expect(layer.mask.bitmap).toBeNull();
    expect(layer.mask.fill).toEqual(DEFAULT_INPAINT_MASK_FILL);
    expect(layer.mask.fill.style).toBe('diagonal');
  });

  it('mints a fresh id when none is given', () => {
    const a = createInpaintMaskLayer('m');
    const b = createInpaintMaskLayer('m');
    expect(a.id).not.toBe(b.id);
  });
});

describe('nextInpaintMaskName', () => {
  it('returns the first free "Inpaint Mask N"', () => {
    expect(nextInpaintMaskName([])).toBe('Inpaint Mask 1');
    expect(nextInpaintMaskName(['Inpaint Mask 1'])).toBe('Inpaint Mask 2');
    // Fills the first gap, ignoring unrelated names.
    expect(nextInpaintMaskName(['Inpaint Mask 1', 'Inpaint Mask 3', 'Layer 2'])).toBe('Inpaint Mask 2');
  });
});

describe('createControlLayer', () => {
  it('builds an empty control layer with the legacy-default ControlNet adapter and transparency on', () => {
    const layer = createControlLayer('Control Layer 1', 'c1');
    expect(layer).toMatchObject({
      id: 'c1',
      isEnabled: true,
      name: 'Control Layer 1',
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(layer.source).toEqual({ bitmap: null, type: 'paint' });
    expect(layer.adapter).toEqual(DEFAULT_CONTROL_ADAPTER);
    // The default adapter's mutable array must be copied, not shared.
    expect(layer.adapter.beginEndStepPct).not.toBe(DEFAULT_CONTROL_ADAPTER.beginEndStepPct);
    expect(layer.adapter.beginEndStepPct).toEqual([0, 0.75]);
  });
});

describe('nextControlLayerName', () => {
  it('returns the first free "Control Layer N" gap', () => {
    expect(nextControlLayerName([])).toBe('Control Layer 1');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 3'])).toBe('Control Layer 2');
    expect(nextControlLayerName(['Control Layer 1', 'Control Layer 2'])).toBe('Control Layer 3');
  });
});

describe('canConvertRasterControl', () => {
  it('is true for image/paint raster layers and control layers, false otherwise', () => {
    expect(canConvertRasterControl(createEmptyPaintLayer('p', 'p'))).toBe(true);
    expect(canConvertRasterControl(createControlLayer('c', 'c'))).toBe(true);
    expect(
      canConvertRasterControl(
        paintLayer('img', { source: { image: { height: 1, imageName: 'i', width: 1 }, type: 'image' } })
      )
    ).toBe(true);
    // Parametric raster sources cannot convert.
    expect(
      canConvertRasterControl(
        paintLayer('txt', {
          source: {
            align: 'left',
            color: '#fff',
            content: 'hi',
            fontFamily: 'x',
            fontSize: 12,
            fontWeight: 400,
            lineHeight: 1,
            type: 'text',
          },
        })
      )
    ).toBe(false);
    expect(canConvertRasterControl(createInpaintMaskLayer('m', 'm'))).toBe(false);
  });
});

describe('fitLayerTransformToBbox', () => {
  it('scales and positions a layer content rect into the bbox', () => {
    const layer = paintLayer('paint', {
      source: { bitmap: { height: 50, imageName: 'paint', width: 100 }, offset: { x: 20, y: 10 }, type: 'paint' },
    });

    expect(
      fitLayerTransformToBbox(layer, { height: 100, width: 200, x: 0, y: 0 }, { height: 512, width: 512, x: 0, y: 0 })
    ).toEqual({ rotation: 0, scaleX: 2, scaleY: 2, x: -40, y: -20 });
  });

  it('returns null for empty content', () => {
    expect(fitLayerTransformToBbox(createEmptyPaintLayer('empty'), { height: 512, width: 512, x: 0, y: 0 })).toBeNull();
  });
});

describe('menu patch helpers', () => {
  it('builds control transparency effect patches', () => {
    const layer = createControlLayer('control', 'c1');

    expect(getControlTransparencyEffectPatch(layer)).toEqual({
      forward: { layerType: 'control', withTransparencyEffect: false },
      inverse: { layerType: 'control', withTransparencyEffect: true },
    });
  });

  it('builds regional prompt and reference image patches', () => {
    const layer = createRegionalGuidanceLayer('region', 0, 'r1');

    expect(getRegionalGuidancePositivePromptPatch(layer)).toEqual({
      forward: { layerType: 'regional_guidance', positivePrompt: '' },
      inverse: { layerType: 'regional_guidance', positivePrompt: null },
    });
    expect(getRegionalGuidanceNegativePromptPatch(layer)).toEqual({
      forward: { layerType: 'regional_guidance', negativePrompt: '' },
      inverse: { layerType: 'regional_guidance', negativePrompt: null },
    });
    expect(getRegionalGuidanceAutoNegativePatch(layer)).toEqual({
      forward: { autoNegative: true, layerType: 'regional_guidance' },
      inverse: { autoNegative: false, layerType: 'regional_guidance' },
    });
    expect(getRegionalGuidanceReferenceImagePatch(layer, 'sdxl').forward.referenceImages).toHaveLength(1);
  });

  it('builds inpaint modifier patches', () => {
    const layer = createInpaintMaskLayer('mask', 'm1');

    expect(getInpaintNoisePatch(layer)).toEqual({
      forward: { layerType: 'inpaint_mask', noiseLevel: 0.25 },
      inverse: { layerType: 'inpaint_mask', noiseLevel: undefined },
    });
    expect(getInpaintDenoiseLimitPatch(layer)).toEqual({
      forward: { denoiseLimit: 0.8, layerType: 'inpaint_mask' },
      inverse: { denoiseLimit: undefined, layerType: 'inpaint_mask' },
    });
  });
});

describe('convertRasterControlLayer', () => {
  it('converts raster→control preserving source/id/name/transform and applying the default adapter', () => {
    const source = { image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' } as const;
    const transform = { rotation: 10, scaleX: 2, scaleY: 2, x: 5, y: 6 };
    const raster = paintLayer('r', { name: 'My Layer', opacity: 0.5, source, transform });
    const converted = convertRasterControlLayer(raster, 'control');
    expect(converted).toMatchObject({
      id: 'r',
      name: 'My Layer',
      opacity: 0.5,
      type: 'control',
      withTransparencyEffect: true,
    });
    if (converted?.type === 'control') {
      expect(converted.source).toEqual(source);
      expect(converted.transform).toEqual(transform);
      expect(converted.adapter).toEqual(DEFAULT_CONTROL_ADAPTER);
    }
  });

  it('converts control→raster preserving the pixel source and round-trips', () => {
    const control = createControlLayer('C', 'c1');
    const withSource = {
      ...control,
      source: { image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' },
    } as CanvasLayerContract;
    const raster = convertRasterControlLayer(withSource, 'raster');
    expect(raster?.type).toBe('raster');
    if (raster?.type === 'raster') {
      expect(raster.source).toEqual({ image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' });
      expect(raster.id).toBe('c1');
    }
    // Round-trip back to control keeps the source.
    const backToControl = raster ? convertRasterControlLayer(raster, 'control') : null;
    expect(backToControl?.type).toBe('control');
    if (backToControl?.type === 'control') {
      expect(backToControl.source).toEqual({ image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' });
    }
  });

  it('returns null when the layer already has the target type or cannot convert', () => {
    expect(convertRasterControlLayer(createControlLayer('c', 'c'), 'control')).toBeNull();
    expect(convertRasterControlLayer(createEmptyPaintLayer('p', 'p'), 'raster')).toBeNull();
    expect(convertRasterControlLayer(createInpaintMaskLayer('m', 'm'), 'control')).toBeNull();
  });
});

describe('createRegionalReferenceImage', () => {
  it('mints a FLUX Redux ref image for a flux base', () => {
    const ref = createRegionalReferenceImage('flux', 'ref-1');
    expect(ref).toEqual({
      config: { image: null, imageInfluence: 'highest', model: null, type: 'flux_redux' },
      id: 'ref-1',
      isEnabled: true,
    });
  });

  it('mints an IP-Adapter ref image for a non-flux base', () => {
    const ref = createRegionalReferenceImage('sdxl', 'ref-2');
    expect(ref.id).toBe('ref-2');
    expect(ref.isEnabled).toBe(true);
    expect(ref.config.type).toBe('ip_adapter');
  });
});

describe('createRegionalGuidanceLayerWithRefImage', () => {
  it('is a regional guidance layer pre-seeded with exactly one empty reference image', () => {
    const layer = createRegionalGuidanceLayerWithRefImage('Region 1', 0, 'sdxl', 'rg-1');
    expect(layer.type).toBe('regional_guidance');
    expect(layer.id).toBe('rg-1');
    expect(layer.referenceImages).toHaveLength(1);
    expect(layer.referenceImages[0]?.config.type).toBe('ip_adapter');
    expect(layer.referenceImages[0]?.config.image).toBeNull();
  });

  it('seeds a FLUX Redux reference image when the base is flux', () => {
    const layer = createRegionalGuidanceLayerWithRefImage('Region 1', 0, 'flux', 'rg-2');
    expect(layer.referenceImages[0]?.config.type).toBe('flux_redux');
  });
});
