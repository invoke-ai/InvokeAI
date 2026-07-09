import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/types';

import { describe, expect, it } from 'vitest';

import type { Rect } from './types';

import { planComposites, planControlComposites, planRegionalMaskComposites } from './compositePlan';

const imageRef = (imageName: string, width = 64, height = 48): CanvasImageRef => ({ height, imageName, width });

const rasterLayer = (
  id: string,
  overrides: Partial<CanvasRasterLayerContractV2> = {}
): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: imageRef(id), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const maskLayer = (id: string): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 1,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

const BBOX: Rect = { height: 200, width: 200, x: 0, y: 0 };

const makeDoc = (
  layers: CanvasLayerContract[],
  overrides: Partial<CanvasDocumentContractV2> = {}
): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 200, width: 200, x: 0, y: 0 },
  height: 300,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 400,
  ...overrides,
});

const keyOf = (doc: CanvasDocumentContractV2, bbox: Rect = BBOX): string => planComposites(doc, bbox).entries[0]!.key;

describe('planComposites — plan shape', () => {
  it('emits a single base-raster entry scoped to the bbox', () => {
    const plan = planComposites(makeDoc([rasterLayer('a')]), BBOX);
    expect(plan.bbox).toEqual(BBOX);
    expect(plan.entries).toHaveLength(1);
    const entry = plan.entries[0]!;
    expect(entry.kind).toBe('base-raster');
    expect(entry.bbox).toEqual(BBOX);
  });

  it('includes only enabled raster (image/paint) layers, preserving z-order', () => {
    const doc = makeDoc([
      rasterLayer('top'),
      maskLayer('mask'),
      rasterLayer('disabled', { isEnabled: false }),
      rasterLayer('bottom', { source: { bitmap: imageRef('paint-bmp'), type: 'paint' } }),
    ]);
    const entry = planComposites(doc, BBOX).entries[0]!;
    expect(entry.layers.map((l) => l.id)).toEqual(['top', 'bottom']);
  });

  it('excludes an inpaint mask (even one with a persisted bitmap) from the base-raster composite', () => {
    const inpaintMask: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'inpaint',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: imageRef('mask-bmp'), fill: { color: '#e07575', style: 'diagonal' } },
      name: 'inpaint',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    };
    const doc = makeDoc([rasterLayer('top'), inpaintMask]);
    const entry = planComposites(doc, BBOX).entries[0]!;
    // Masks feed the (next task's) mask composite, never the image composite.
    expect(entry.layers.map((l) => l.id)).toEqual(['top']);
  });

  it('derives sourceRef + content-sized contentSize/contentOffset for image and paint layers', () => {
    const doc = makeDoc([
      rasterLayer('img', { source: { image: imageRef('pic', 128, 96), type: 'image' } }),
      rasterLayer('paint', { source: { bitmap: imageRef('bmp', 200, 150), offset: { x: 40, y: 25 }, type: 'paint' } }),
    ]);
    const [img, paint] = planComposites(doc, BBOX).entries[0]!.layers;
    expect(img!.sourceRef).toBe('image:pic');
    expect(img!.contentSize).toEqual({ height: 96, width: 128 });
    expect(img!.contentOffset).toEqual({ x: 0, y: 0 });
    expect(paint!.sourceRef).toBe('paint:bmp');
    // Paint layers are content-sized: the persisted bitmap dims placed at its offset.
    expect(paint!.contentSize).toEqual({ height: 150, width: 200 });
    expect(paint!.contentOffset).toEqual({ x: 40, y: 25 });
  });

  it('excludes an empty (bitmap: null) paint layer so it cannot force outpaint', () => {
    // An auto-created paint layer left blank (e.g. its stroke was undone) carries
    // no pixels; including it would inject a doc-sized transparent rect that reads
    // as outpaint. It must not appear among the base-raster contributors.
    const doc = makeDoc([
      rasterLayer('img', { source: { image: imageRef('pic'), type: 'image' } }),
      rasterLayer('blank', { source: { bitmap: null, type: 'paint' } }),
    ]);
    const entry = planComposites(doc, BBOX).entries[0]!;
    expect(entry.layers.map((l) => l.id)).toEqual(['img']);
  });

  it('emits an empty layer list when the only paint layer is unpainted', () => {
    const doc = makeDoc([rasterLayer('blank', { source: { bitmap: null, type: 'paint' } })]);
    expect(planComposites(doc, BBOX).entries[0]!.layers).toEqual([]);
  });
});

describe('planComposites — key stability', () => {
  it('produces byte-identical keys for structurally identical documents', () => {
    const build = () => makeDoc([rasterLayer('a'), rasterLayer('b', { opacity: 0.5 })]);
    expect(keyOf(build())).toBe(keyOf(build()));
  });
});

describe('planComposites — key sensitivity', () => {
  it('changes the key when layers are reordered', () => {
    const a = makeDoc([rasterLayer('a'), rasterLayer('b')]);
    const b = makeDoc([rasterLayer('b'), rasterLayer('a')]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });

  it('changes the key when a layer opacity changes', () => {
    const a = makeDoc([rasterLayer('a', { opacity: 1 })]);
    const b = makeDoc([rasterLayer('a', { opacity: 0.5 })]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });

  it('changes the key when a layer transform changes', () => {
    const a = makeDoc([rasterLayer('a')]);
    const b = makeDoc([rasterLayer('a', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 0 } })]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });

  it('changes the key when a layer blend mode changes', () => {
    const a = makeDoc([rasterLayer('a', { blendMode: 'normal' })]);
    const b = makeDoc([rasterLayer('a', { blendMode: 'multiply' })]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });

  it('changes the key when a layer source is swapped', () => {
    const a = makeDoc([rasterLayer('a', { source: { image: imageRef('cat'), type: 'image' } })]);
    const b = makeDoc([rasterLayer('a', { source: { image: imageRef('dog'), type: 'image' } })]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });

  it('changes the key when the bbox moves', () => {
    const doc = makeDoc([rasterLayer('a')]);
    expect(keyOf(doc, { height: 200, width: 200, x: 0, y: 0 })).not.toBe(
      keyOf(doc, { height: 200, width: 200, x: 10, y: 0 })
    );
  });

  it('changes the key when a layer is toggled off', () => {
    const a = makeDoc([rasterLayer('a')]);
    const b = makeDoc([rasterLayer('a', { isEnabled: false })]);
    expect(keyOf(a)).not.toBe(keyOf(b));
  });
});

describe('planComposites — key insensitivity to irrelevant changes', () => {
  it('ignores selectedLayerId', () => {
    const a = makeDoc([rasterLayer('a')], { selectedLayerId: null });
    const b = makeDoc([rasterLayer('a')], { selectedLayerId: 'a' });
    expect(keyOf(a)).toBe(keyOf(b));
  });

  it('ignores the document background', () => {
    const a = makeDoc([rasterLayer('a')], { background: 'transparent' });
    const b = makeDoc([rasterLayer('a')], { background: { color: '#123456' } });
    expect(keyOf(a)).toBe(keyOf(b));
  });

  it('ignores edits to a disabled layer', () => {
    const a = makeDoc([rasterLayer('a'), rasterLayer('off', { isEnabled: false, opacity: 1 })]);
    const b = makeDoc([rasterLayer('a'), rasterLayer('off', { isEnabled: false, opacity: 0.2, blendMode: 'screen' })]);
    expect(keyOf(a)).toBe(keyOf(b));
  });
});

const inpaintMask = (
  id: string,
  overrides: Partial<{
    bitmap: CanvasImageRef | null;
    noiseLevel: number;
    denoiseLimit: number;
    isEnabled: boolean;
    offset: { x: number; y: number };
  }> = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  denoiseLimit: overrides.denoiseLimit,
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  mask: {
    bitmap: 'bitmap' in overrides ? overrides.bitmap! : imageRef(`${id}-bmp`, 100, 80),
    fill: { color: '#ff0000', style: 'solid' },
    ...(overrides.offset ? { offset: overrides.offset } : {}),
  },
  name: id,
  noiseLevel: overrides.noiseLevel,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const entryOfKind = (doc: CanvasDocumentContractV2, kind: string) =>
  planComposites(doc, BBOX).entries.find((e) => e.kind === kind);

describe('planComposites — inpaint-mask entries', () => {
  it('emits no mask entries when there are no active inpaint masks', () => {
    const kinds = planComposites(makeDoc([rasterLayer('a')]), BBOX).entries.map((e) => e.kind);
    expect(kinds).toEqual(['base-raster']);
  });

  it('emits an inpaint-mask entry for enabled masks with a persisted bitmap', () => {
    const entry = entryOfKind(makeDoc([rasterLayer('a'), inpaintMask('m1')]), 'inpaint-mask');
    expect(entry).toBeDefined();
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['m1']);
    expect(entry!.maskLayers![0]!.sourceRef).toBe('mask:m1-bmp');
    expect(entry!.maskLayers![0]!.contentSize).toEqual({ height: 80, width: 100 });
  });

  it('excludes disabled masks and empty (bitmap: null) masks', () => {
    const doc = makeDoc([
      inpaintMask('on'),
      inpaintMask('off', { isEnabled: false }),
      inpaintMask('empty', { bitmap: null }),
    ]);
    const entry = entryOfKind(doc, 'inpaint-mask');
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['on']);
  });

  it('resolves an undefined denoiseLimit to the legacy default (1.0)', () => {
    const entry = entryOfKind(makeDoc([inpaintMask('m1')]), 'inpaint-mask');
    expect(entry!.maskLayers![0]!.attributeValue).toBe(1);
  });

  it('uses the layer denoiseLimit when defined', () => {
    const entry = entryOfKind(makeDoc([inpaintMask('m1', { denoiseLimit: 0.4 })]), 'inpaint-mask');
    expect(entry!.maskLayers![0]!.attributeValue).toBe(0.4);
  });

  it('unions multiple masks into the denoise-limit entry', () => {
    const doc = makeDoc([inpaintMask('a', { denoiseLimit: 0.3 }), inpaintMask('b')]);
    const entry = entryOfKind(doc, 'inpaint-mask');
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['a', 'b']);
  });
});

describe('planComposites — noise-mask entries', () => {
  it('omits the noise-mask entry when no mask defines a noiseLevel (undefined is NOT 0)', () => {
    const kinds = planComposites(makeDoc([inpaintMask('m1')]), BBOX).entries.map((e) => e.kind);
    expect(kinds).toEqual(['base-raster', 'inpaint-mask']);
  });

  it('emits a noise-mask entry only for masks with a defined noiseLevel', () => {
    const doc = makeDoc([inpaintMask('withNoise', { noiseLevel: 0.15 }), inpaintMask('noNoise')]);
    const entry = entryOfKind(doc, 'noise-mask');
    expect(entry).toBeDefined();
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['withNoise']);
    expect(entry!.maskLayers![0]!.attributeValue).toBe(0.15);
  });

  it('treats noiseLevel 0 as defined (included), distinct from undefined (excluded)', () => {
    const entry = entryOfKind(makeDoc([inpaintMask('m1', { noiseLevel: 0 })]), 'noise-mask');
    expect(entry).toBeDefined();
    expect(entry!.maskLayers![0]!.attributeValue).toBe(0);
  });
});

describe('planComposites — mask entry keys', () => {
  const maskKeyOf = (doc: CanvasDocumentContractV2, kind: string): string | undefined => entryOfKind(doc, kind)?.key;

  it('is stable for identical documents', () => {
    const a = makeDoc([inpaintMask('m1', { denoiseLimit: 0.5 })]);
    const b = makeDoc([inpaintMask('m1', { denoiseLimit: 0.5 })]);
    expect(maskKeyOf(a, 'inpaint-mask')).toBe(maskKeyOf(b, 'inpaint-mask'));
  });

  it('changes when the attribute value changes', () => {
    const a = makeDoc([inpaintMask('m1', { denoiseLimit: 0.5 })]);
    const b = makeDoc([inpaintMask('m1', { denoiseLimit: 0.6 })]);
    expect(maskKeyOf(a, 'inpaint-mask')).not.toBe(maskKeyOf(b, 'inpaint-mask'));
  });

  it('changes when the mask bitmap changes', () => {
    const a = makeDoc([inpaintMask('m1')]);
    const b = makeDoc([inpaintMask('m1', { bitmap: imageRef('other', 100, 80) })]);
    expect(maskKeyOf(a, 'inpaint-mask')).not.toBe(maskKeyOf(b, 'inpaint-mask'));
  });
});

const controlLayer = (
  id: string,
  overrides: Partial<{
    source: CanvasControlLayerContract['source'];
    isEnabled: boolean;
    withTransparencyEffect: boolean;
    opacity: number;
    blendMode: CanvasLayerContract['blendMode'];
  }> = {}
): CanvasLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: overrides.blendMode ?? 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isLocked: false,
  name: id,
  opacity: overrides.opacity ?? 1,
  source: overrides.source ?? { image: imageRef(id, 64, 48), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: overrides.withTransparencyEffect ?? false,
});

describe('planComposites — control-layer exclusion', () => {
  it('never includes control layers in the base-raster entry', () => {
    const plan = planComposites(makeDoc([rasterLayer('r'), controlLayer('c')]), BBOX);
    const base = plan.entries.find((entry) => entry.kind === 'base-raster');
    expect(base?.layers.map((layer) => layer.id)).toEqual(['r']);
  });

  it('emits no control-layer entries from planComposites (they live in planControlComposites)', () => {
    const plan = planComposites(makeDoc([controlLayer('c')]), BBOX);
    expect(plan.entries.some((entry) => entry.kind === 'control-layer')).toBe(false);
  });
});

describe('planControlComposites', () => {
  it('emits one separate entry per enabled control layer with content', () => {
    const composites = planControlComposites(makeDoc([controlLayer('c1'), controlLayer('c2')]), BBOX);
    expect(composites.map((composite) => composite.layerId)).toEqual(['c1', 'c2']);
    for (const { entry } of composites) {
      expect(entry.kind).toBe('control-layer');
      expect(entry.bbox).toEqual(BBOX);
      expect(entry.layers).toHaveLength(1);
      // Each control layer is composited SEPARATELY, never blended with others.
      expect(entry.layerId).toBe(entry.layers[0]!.id);
    }
    // Distinct keys per layer.
    expect(composites[0]!.entry.key).not.toBe(composites[1]!.entry.key);
  });

  it('excludes disabled control layers and empty paint-source control layers', () => {
    const composites = planControlComposites(
      makeDoc([
        controlLayer('enabled'),
        controlLayer('disabled', { isEnabled: false }),
        controlLayer('empty', { source: { bitmap: null, type: 'paint' } }),
      ]),
      BBOX
    );
    expect(composites.map((composite) => composite.layerId)).toEqual(['enabled']);
  });

  it('ignores raster/mask layers entirely', () => {
    const composites = planControlComposites(makeDoc([rasterLayer('r'), inpaintMask('m'), controlLayer('c')]), BBOX);
    expect(composites.map((composite) => composite.layerId)).toEqual(['c']);
  });

  it('forces opacity 1 / normal blend so display tweaks and the transparency effect never churn the key', () => {
    const a = planControlComposites(makeDoc([controlLayer('c')]), BBOX);
    const b = planControlComposites(
      makeDoc([controlLayer('c', { opacity: 0.3, blendMode: 'multiply', withTransparencyEffect: true })]),
      BBOX
    );
    expect(a[0]!.entry.key).toBe(b[0]!.entry.key);
    expect(a[0]!.entry.layers[0]!.opacity).toBe(1);
    expect(a[0]!.entry.layers[0]!.blendMode).toBe('normal');
  });

  it('changes the key when the control source pixels change', () => {
    const a = planControlComposites(makeDoc([controlLayer('c')]), BBOX);
    const b = planControlComposites(
      makeDoc([controlLayer('c', { source: { image: imageRef('other', 64, 48), type: 'image' } })]),
      BBOX
    );
    expect(a[0]!.entry.key).not.toBe(b[0]!.entry.key);
  });
});

const regionalLayer = (
  id: string,
  overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}
): CanvasLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: imageRef(`${id}-mask`), fill: { color: '#ff0000', style: 'solid' } },
  name: id,
  negativePrompt: null,
  opacity: 0.5,
  positivePrompt: 'a cat',
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
  ...overrides,
});

describe('planRegionalMaskComposites', () => {
  it('emits one alpha-mask entry per enabled region WITH mask content', () => {
    const entries = planRegionalMaskComposites(makeDoc([regionalLayer('a'), regionalLayer('b')]), BBOX);
    expect(entries).toHaveLength(2);
    expect(entries[0]!.entry.kind).toBe('regional-mask');
    expect(entries[0]!.layerId).toBe('a');
    // Composited at opacity 1 / normal (display opacity 0.5 must NOT alter the mask alpha sent to the backend).
    expect(entries[0]!.entry.layers[0]!.opacity).toBe(1);
    expect(entries[0]!.entry.layers[0]!.blendMode).toBe('normal');
  });

  it('skips regions with no mask bitmap and disabled regions', () => {
    const entries = planRegionalMaskComposites(
      makeDoc([
        regionalLayer('empty', { mask: { bitmap: null, fill: { color: '#f00', style: 'solid' } } }),
        regionalLayer('off', { isEnabled: false }),
        regionalLayer('ok'),
      ]),
      BBOX
    );
    expect(entries.map((e) => e.layerId)).toEqual(['ok']);
  });

  it('keys are stable per region+bbox and change with the mask source', () => {
    const a = planRegionalMaskComposites(makeDoc([regionalLayer('a')]), BBOX);
    const b = planRegionalMaskComposites(
      makeDoc([
        regionalLayer('a', { mask: { bitmap: imageRef('other-mask'), fill: { color: '#f00', style: 'solid' } } }),
      ]),
      BBOX
    );
    expect(a[0]!.entry.key).not.toBe(b[0]!.entry.key);
  });
});

describe('planComposites — raster adjustments in the base key', () => {
  it('folds a non-identity adjustment into the base-raster entry (and its key)', () => {
    const plain = rasterLayer('a');
    const adjusted = rasterLayer('a', { adjustments: { brightness: 0.5, contrast: 0, saturation: 0 } });
    const plan = planComposites(makeDoc([adjusted]), BBOX);
    const ref = plan.entries[0]!.layers[0]!;
    expect(ref.adjustments).toEqual({ brightness: 0.5, contrast: 0, saturation: 0 });
    // A changed adjustment changes the entry key so the executor re-composites/uploads.
    expect(keyOf(makeDoc([plain]))).not.toBe(keyOf(makeDoc([adjusted])));
  });

  it('ignores an identity adjustment (no key churn)', () => {
    const plain = rasterLayer('a');
    const identityAdj = rasterLayer('a', { adjustments: { brightness: 0, contrast: 0, saturation: 0 } });
    expect(plan0(plain)).toBe(plan0(identityAdj));
  });
});

const plan0 = (layer: CanvasLayerContract): string => planComposites(makeDoc([layer]), BBOX).entries[0]!.key;
