import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { createControlLayer } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import {
  canConvertRasterControl,
  canMergeLayerDown,
  convertRasterToControl,
  convertRasterToInpaintMask,
  convertRasterToRegionalGuidance,
  convertRasterControlLayer,
  copyControlToInpaintMask,
  copyControlToRaster,
  copyControlToRegionalGuidance,
  copyMaskToRegionalGuidance,
  copyRasterToControl,
  copyRasterToInpaintMask,
  copyRasterToRegionalGuidance,
  copyRegionalGuidanceToInpaintMask,
  createAdjustmentEntry,
  createEmptyPaintLayer,
  createInpaintMaskFromImage,
  createInpaintMaskLayer,
  createRegionalGuidanceFromImage,
  createRegionalGuidanceLayerWithRefImage,
  createRegionalReferenceImage,
  DEFAULT_CONTROL_ADAPTER,
  CONTROL_ADAPTER_DEFAULTS,
  fitLayerTransformToBbox,
  isAutoControlLayerName,
  isMergeableRasterLayer,
  nextControlTypeLayerName,
} from './layerOps';

seedArchitectureCapabilities();

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

  it('mints a UUID id, so a lock or presence marker can name the layer whoever created it', () => {
    expect(createEmptyPaintLayer('a').id).toMatch(
      /^layer-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    );
  });
});

describe('createAdjustmentEntry', () => {
  it('starts noise as visible film grain: monochrome, fading toward black and white, per-pixel, 10%, fresh seed', () => {
    const first = createAdjustmentEntry('noise');
    const second = createAdjustmentEntry('noise');
    expect(first).toMatchObject({
      amount: 0.1,
      distribution: 'gaussian',
      grain: 'monochrome',
      isEnabled: true,
      response: 'midtones',
      size: 1,
      type: 'noise',
    });
    for (const entry of [first, second]) {
      const { seed } = entry as Extract<typeof entry, { type: 'noise' }>;
      expect(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff).toBe(true);
    }
    expect(first.id).not.toBe(second.id);
  });
});

describe('isMergeableRasterLayer', () => {
  it.each<[string, CanvasLayerContract, boolean]>([
    ['accepts an enabled paint raster layer', paintLayer('p'), true],
    ['accepts an enabled image raster layer', imageLayer('i'), true],
    ['rejects a disabled layer', paintLayer('p', { isEnabled: false }), false],
    ['rejects a non-raster layer', maskLayer('m'), false],
    ['rejects a locked paint layer', paintLayer('p', { isLocked: true }), false],
    ['rejects a locked image layer', { ...imageLayer('i'), isLocked: true }, false],
  ])('%s', (_, layer, expected) => {
    expect(isMergeableRasterLayer(layer)).toBe(expected);
  });
});

describe('canMergeLayerDown', () => {
  const documentOf = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    stacks: stacksFrom(layers),
    selectedLayerId: null,
    version: 3,
    width: 100,
  });
  const document = documentOf([paintLayer('top'), imageLayer('mid'), maskLayer('bottom')]);

  it.each<[string, CanvasDocumentContractV3, string, boolean, boolean]>([
    ['requires an engine (merge is pixel work)', document, 'top', false, false],
    ['allows merging when the layer and the one below are both mergeable', document, 'top', true, true],
    ['disallows merging into a non-mergeable below layer', document, 'mid', true, false],
    ['disallows merging the bottom-most layer (nothing below)', document, 'bottom', true, false],
    [
      'disallows merging when the upper layer is locked',
      documentOf([paintLayer('top', { isLocked: true }), imageLayer('mid'), maskLayer('bottom')]),
      'top',
      true,
      false,
    ],
    [
      'disallows merging when the below layer is locked',
      documentOf([paintLayer('top'), { ...imageLayer('mid'), isLocked: true }, maskLayer('bottom')]),
      'top',
      true,
      false,
    ],
  ])('%s', (_, doc, layerId, hasEngine, expected) => {
    expect(canMergeLayerDown(doc, layerId, hasEngine)).toBe(expected);
  });
});

describe('image-backed mask layer factories', () => {
  const image = { height: 13, imageName: 'selected-object.png', width: 17 };
  const rect = { height: 13, width: 17, x: -4, y: 9 };

  it('builds a complete inpaint mask contract with the image at the world rect', () => {
    expect(
      createInpaintMaskFromImage({
        fill: { color: '#123456', style: 'crosshatch' },
        id: 'mask-result',
        image,
        name: 'Inpaint Mask 2',
        rect,
      })
    ).toEqual({
      blendMode: 'normal',
      id: 'mask-result',
      isEnabled: true,
      isLocked: false,
      mask: {
        bitmap: image,
        fill: { color: '#123456', style: 'crosshatch' },
        offset: { x: -4, y: 9 },
      },
      name: 'Inpaint Mask 2',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    });
  });

  it('builds a complete regional-guidance contract with preserved defaults', () => {
    expect(
      createRegionalGuidanceFromImage({
        fill: { color: '#83d683', style: 'solid' },
        id: 'regional-result',
        image,
        name: 'Regional Guidance 3',
        rect,
      })
    ).toEqual({
      autoNegative: false,
      blendMode: 'normal',
      id: 'regional-result',
      isEnabled: true,
      isLocked: false,
      mask: {
        bitmap: image,
        fill: { color: '#83d683', style: 'solid' },
        offset: { x: -4, y: 9 },
      },
      name: 'Regional Guidance 3',
      negativePrompt: null,
      opacity: 0.5,
      positivePrompt: null,
      referenceImages: [],
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'regional_guidance',
    });
  });
});

describe('control layer naming by type', () => {
  it('numbers a typed layer after the other layers of its type, filling the first gap', () => {
    expect(nextControlTypeLayerName('Depth', [])).toBe('Depth 1');
    expect(nextControlTypeLayerName('Depth', ['Depth 1', 'Depth 3', 'Pose 2', 'Control Layer 2'])).toBe('Depth 2');
    expect(nextControlTypeLayerName('Line art (v2)', ['Line art (v2) 1'])).toBe('Line art (v2) 2');
  });

  it('renames only names the app gave the layer', () => {
    const labels = ['Depth', 'Soft edges'];
    expect(isAutoControlLayerName('Control Layer 4', labels)).toBe(true);
    expect(isAutoControlLayerName('Soft edges 2', labels)).toBe(true);
    expect(isAutoControlLayerName('Depth', labels)).toBe(false);
    expect(isAutoControlLayerName('My depth pass', labels)).toBe(false);
    expect(isAutoControlLayerName('Tile 1', labels)).toBe(false);
  });
});

describe('control adapter defaults', () => {
  it('centralizes legacy defaults for every adapter kind', () => {
    expect(CONTROL_ADAPTER_DEFAULTS).toEqual({
      anima_lllite: { beginEndStepPct: [0, 1], controlMode: null, kind: 'anima_lllite', model: null, weight: 1 },
      control_lora: { beginEndStepPct: [0, 1], controlMode: null, kind: 'control_lora', model: null, weight: 0.75 },
      controlnet: {
        beginEndStepPct: [0, 0.75],
        controlMode: 'balanced',
        kind: 'controlnet',
        model: null,
        weight: 0.75,
      },
      flux2_control: { beginEndStepPct: [0, 1], controlMode: null, kind: 'flux2_control', model: null, weight: 0.75 },
      qwen_image_control: {
        beginEndStepPct: [0, 1],
        controlMode: null,
        kind: 'qwen_image_control',
        model: null,
        weight: 0.85,
      },
      t2i_adapter: { beginEndStepPct: [0, 1], controlMode: null, kind: 't2i_adapter', model: null, weight: 1 },
      z_image_control: {
        beginEndStepPct: [0, 1],
        controlMode: null,
        kind: 'z_image_control',
        model: null,
        weight: 0.75,
      },
    });
  });
});

describe('canConvertRasterControl', () => {
  it('is true for image/paint raster layers and control layers, false otherwise', () => {
    expect(canConvertRasterControl(createEmptyPaintLayer('p', 'p'))).toBe(true);
    expect(canConvertRasterControl(createControlLayer({ id: 'c', name: 'c' }))).toBe(true);
    expect(
      canConvertRasterControl(
        paintLayer('img', { source: { image: { height: 1, imageName: 'i', width: 1 }, type: 'image' } })
      )
    ).toBe(true);
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

  it('contains and centers landscape content without stretching', () => {
    const layer = paintLayer('landscape', {
      source: {
        bitmap: { height: 100, imageName: 'landscape', width: 200 },
        offset: { x: 10, y: 20 },
        type: 'paint',
      },
    });

    expect(fitLayerTransformToBbox(layer, { height: 100, width: 100, x: 0, y: 0 })).toEqual({
      rotation: 0,
      scaleX: 0.5,
      scaleY: 0.5,
      x: -5,
      y: 15,
    });
  });

  it.each([
    ['cover', { scaleX: 1, scaleY: 1, x: -60, y: -20 }],
    ['fill', { scaleX: 0.5, scaleY: 1, x: -5, y: -20 }],
  ] as const)('fits offset content with %s', (mode, expected) => {
    const layer = paintLayer('fit', {
      source: { type: 'paint', bitmap: { imageName: 'fit.png', width: 200, height: 100 }, offset: { x: 10, y: 20 } },
    });
    const bbox = { x: 0, y: 0, width: 100, height: 100 };
    expect(fitLayerTransformToBbox(layer, bbox, bbox, mode)).toEqual({ rotation: 0, ...expected });
  });
});

describe('convertRasterControlLayer', () => {
  it('converts raster→control preserving source/id/name/transform and applying the default adapter', () => {
    const source = { image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' } as const;
    const transform = { rotation: 10, scaleX: 2, scaleY: 2, x: 5, y: 6 };
    const raster = paintLayer('r', { colorLabel: 'red', name: 'My Layer', opacity: 0.5, source, transform });
    const converted = convertRasterControlLayer(raster, 'control');
    expect(converted).toMatchObject({
      colorLabel: 'red',
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
    const control = createControlLayer({ id: 'c1', name: 'C' });
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
    const backToControl = raster ? convertRasterControlLayer(raster, 'control') : null;
    expect(backToControl?.type).toBe('control');
    if (backToControl?.type === 'control') {
      expect(backToControl.source).toEqual({ image: { height: 48, imageName: 'pic', width: 64 }, type: 'image' });
    }
  });
});

describe('layer copy and conversion parity', () => {
  const transform = { rotation: 10, scaleX: 2, scaleY: 3, x: 4, y: 5 };
  const raster = {
    ...createEmptyPaintLayer('Raster', 'r1'),
    blendMode: 'multiply' as const,
    isEnabled: false,
    isLocked: true,
    opacity: 0.4,
    source: {
      bitmap: { height: 20, imageName: 'paint-bitmap', width: 30 },
      offset: { x: -3, y: 7 },
      type: 'paint' as const,
    },
    transform,
  };

  it('copies and converts pixel-backed raster layers without baking their source', () => {
    const controlCopy = copyRasterToControl(raster, 'c-copy');
    expect(controlCopy).toMatchObject({
      id: 'c-copy',
      isEnabled: false,
      isLocked: true,
      name: 'Raster copy',
      opacity: 0.4,
      source: raster.source,
      transform,
      type: 'control',
      withTransparencyEffect: true,
    });
    expect(copyRasterToInpaintMask(raster, 'm-copy')).toMatchObject({
      id: 'm-copy',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      name: 'Raster copy',
      type: 'inpaint_mask',
    });
    expect(copyRasterToRegionalGuidance(raster, 'rg-copy')).toMatchObject({
      autoNegative: false,
      id: 'rg-copy',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      name: 'Raster copy',
      negativePrompt: null,
      positivePrompt: null,
      referenceImages: [],
      type: 'regional_guidance',
    });

    expect(convertRasterToControl(raster)).toMatchObject({ id: 'r1', source: raster.source, type: 'control' });
    expect(convertRasterToInpaintMask(raster)).toMatchObject({
      id: 'r1',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      type: 'inpaint_mask',
    });
    expect(convertRasterToRegionalGuidance(raster)).toMatchObject({
      id: 'r1',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      type: 'regional_guidance',
    });
  });

  it('copies control layers to pixel and mask-backed destinations', () => {
    const control = { ...createControlLayer({ id: 'c1', name: 'Control' }), source: raster.source, transform };

    expect(copyControlToRaster(control, 'r-copy')).toMatchObject({
      id: 'r-copy',
      name: 'Control copy',
      source: raster.source,
      transform,
      type: 'raster',
    });
    expect(copyControlToInpaintMask(control, 'm-copy')).toMatchObject({
      id: 'm-copy',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      type: 'inpaint_mask',
    });
    expect(copyControlToRegionalGuidance(control, 'rg-copy')).toMatchObject({
      id: 'rg-copy',
      mask: { bitmap: raster.source.bitmap, offset: raster.source.offset },
      type: 'regional_guidance',
    });
  });

  it('copies between mask layer types while preserving bitmap, offset, and fill', () => {
    const mask = {
      ...createInpaintMaskLayer('Mask', 'm1'),
      mask: {
        bitmap: { height: 12, imageName: 'mask-bitmap', width: 13 },
        fill: { color: '#123456', style: 'crosshatch' as const },
        offset: { x: 8, y: -2 },
      },
      transform,
    };
    const regional = copyMaskToRegionalGuidance(mask, 'rg-copy');

    expect(regional).toMatchObject({ id: 'rg-copy', mask: mask.mask, name: 'Mask copy', type: 'regional_guidance' });
    expect(regional?.mask).not.toBe(mask.mask);
    expect(copyRegionalGuidanceToInpaintMask(regional!, 'm-copy')).toMatchObject({
      id: 'm-copy',
      mask: mask.mask,
      name: 'Mask copy copy',
      type: 'inpaint_mask',
    });
  });

  it('rejects parametric raster sources that cannot be represented as masks or controls', () => {
    const text = {
      ...raster,
      source: {
        align: 'left' as const,
        color: '#fff',
        content: 'hello',
        fontFamily: 'Inter',
        fontSize: 32,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text' as const,
      },
    };

    expect(copyRasterToControl(text, 'copy')).toBeNull();
    expect(copyRasterToInpaintMask(text, 'copy')).toBeNull();
    expect(copyRasterToRegionalGuidance(text, 'copy')).toBeNull();
    expect(convertRasterToControl(text)).toBeNull();
    expect(convertRasterToInpaintMask(text)).toBeNull();
    expect(convertRasterToRegionalGuidance(text)).toBeNull();
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
});

describe('createRegionalGuidanceLayerWithRefImage', () => {
  it('is a regional guidance layer pre-seeded with exactly one empty reference image', () => {
    const layer = createRegionalGuidanceLayerWithRefImage('Region 1', undefined, 'sdxl', 'rg-1');
    expect(layer.type).toBe('regional_guidance');
    expect(layer.id).toBe('rg-1');
    expect(layer.referenceImages).toHaveLength(1);
    expect(layer.referenceImages[0]?.config.type).toBe('ip_adapter');
    expect(layer.referenceImages[0]?.config.image).toBeNull();
  });
});
