import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/contracts';

import { groupContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import type { Rect } from './generationContracts';

import {
  getBaseRasterContentBounds,
  planComposites,
  planControlComposites,
  planRegionalMaskComposites,
} from './generationCompositePlan';

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

const textLayer = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 =>
  rasterLayer(id, {
    source: {
      align: 'left',
      color: '#ffffff',
      content: 'hello',
      fontFamily: 'Inter',
      fontSize: 20,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text',
    },
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
  overrides: Partial<CanvasDocumentContractV3> = {}
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 200, width: 200, x: 0, y: 0 },
  height: 300,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: 400,
  ...overrides,
});

const keyOf = (doc: CanvasDocumentContractV3, bbox: Rect = BBOX): string => planComposites(doc, bbox).entries[0]!.key;

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

  it('includes committed text with its estimated content rect and complete source identity', () => {
    const doc = makeDoc([textLayer('text')]);
    const entry = planComposites(doc, BBOX).entries[0]!;
    expect(entry.layers.map((layer) => layer.id)).toEqual(['text']);
    expect(entry.layers[0]!.sourceRef).toContain('text:');
    expect(entry.layers[0]!.contentSize).toEqual({ height: 24, width: 60 });

    const edited = makeDoc([
      textLayer('text', {
        source: {
          align: 'left',
          color: '#ffffff',
          content: 'edited text',
          fontFamily: 'Inter',
          fontSize: 20,
          fontWeight: 400,
          lineHeight: 1.2,
          type: 'text',
        },
      }),
    ]);
    expect(planComposites(edited, BBOX).entries[0]!.key).not.toBe(entry.key);
  });

  it('excludes an empty (bitmap: null) paint layer so it cannot force outpaint', () => {
    // Exclude blank paint layers: their transparent document-sized bounds would falsely imply outpaint.
    const doc = makeDoc([
      rasterLayer('img', { source: { image: imageRef('pic'), type: 'image' } }),
      rasterLayer('blank', { source: { bitmap: null, type: 'paint' } }),
    ]);
    const entry = planComposites(doc, BBOX).entries[0]!;
    expect(entry.layers.map((l) => l.id)).toEqual(['img']);
  });
});

describe('planComposites — key stability', () => {
  it('produces byte-identical keys for structurally identical documents', () => {
    const build = () => makeDoc([rasterLayer('a'), rasterLayer('b', { opacity: 0.5 })]);
    expect(keyOf(build())).toBe(keyOf(build()));
  });
});

describe('planComposites — key sensitivity', () => {
  it.each<[string, CanvasDocumentContractV3, CanvasDocumentContractV3]>([
    [
      'layers are reordered',
      makeDoc([rasterLayer('a'), rasterLayer('b')]),
      makeDoc([rasterLayer('b'), rasterLayer('a')]),
    ],
    [
      'a layer opacity changes',
      makeDoc([rasterLayer('a', { opacity: 1 })]),
      makeDoc([rasterLayer('a', { opacity: 0.5 })]),
    ],
    [
      'a layer transform changes',
      makeDoc([rasterLayer('a')]),
      makeDoc([rasterLayer('a', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 0 } })]),
    ],
    [
      'a layer blend mode changes',
      makeDoc([rasterLayer('a', { blendMode: 'normal' })]),
      makeDoc([rasterLayer('a', { blendMode: 'multiply' })]),
    ],
    [
      'a layer source is swapped',
      makeDoc([rasterLayer('a', { source: { image: imageRef('cat'), type: 'image' } })]),
      makeDoc([rasterLayer('a', { source: { image: imageRef('dog'), type: 'image' } })]),
    ],
    ['a layer is toggled off', makeDoc([rasterLayer('a')]), makeDoc([rasterLayer('a', { isEnabled: false })])],
  ])('changes the key when %s', (_label, before, after) => {
    expect(keyOf(before)).not.toBe(keyOf(after));
  });
});

describe('planComposites — key insensitivity to irrelevant changes', () => {
  it.each<[string, CanvasDocumentContractV3, CanvasDocumentContractV3]>([
    [
      'selectedLayerId',
      makeDoc([rasterLayer('a')], { selectedLayerId: null }),
      makeDoc([rasterLayer('a')], { selectedLayerId: 'a' }),
    ],
    [
      'the document background',
      makeDoc([rasterLayer('a')], { background: 'transparent' }),
      makeDoc([rasterLayer('a')], { background: { color: '#123456' } }),
    ],
    [
      'edits to a disabled layer',
      makeDoc([rasterLayer('a'), rasterLayer('off', { isEnabled: false, opacity: 1 })]),
      makeDoc([rasterLayer('a'), rasterLayer('off', { isEnabled: false, opacity: 0.2, blendMode: 'screen' })]),
    ],
  ])('ignores %s', (_label, before, after) => {
    expect(keyOf(before)).toBe(keyOf(after));
  });
});

const inpaintMask = (
  id: string,
  overrides: Partial<{
    bitmap: CanvasImageRef | null;
    noiseLevel: number;
    noiseEnabled: boolean;
    denoiseLimit: number;
    denoiseEnabled: boolean;
    isEnabled: boolean;
    isHidden: boolean;
    offset: { x: number; y: number };
  }> = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  ...(overrides.denoiseLimit === undefined
    ? {}
    : { denoise: { isEnabled: overrides.denoiseEnabled ?? true, limit: overrides.denoiseLimit } }),
  id,
  isEnabled: overrides.isEnabled ?? true,
  isHidden: overrides.isHidden,
  isLocked: false,
  mask: {
    bitmap: 'bitmap' in overrides ? overrides.bitmap! : imageRef(`${id}-bmp`, 100, 80),
    fill: { color: '#ff0000', style: 'solid' },
    ...(overrides.offset ? { offset: overrides.offset } : {}),
  },
  name: id,
  ...(overrides.noiseLevel === undefined
    ? {}
    : { noise: { isEnabled: overrides.noiseEnabled ?? true, level: overrides.noiseLevel } }),
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const entryOfKind = (doc: CanvasDocumentContractV3, kind: string) =>
  planComposites(doc, BBOX).entries.find((e) => e.kind === kind);

describe('planComposites — inpaint-mask entries', () => {
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

  it('INCLUDES hidden masks — hiding is display-only and must not change the image', () => {
    const doc = makeDoc([inpaintMask('shown'), inpaintMask('hidden', { isHidden: true })]);
    const entry = entryOfKind(doc, 'inpaint-mask');
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['shown', 'hidden']);
  });

  it("unions a regenerate-region raster's own alpha into the inpaint mask at full denoise", () => {
    const region = { fill: { color: '#e07575', style: 'diagonal' as const }, isEnabled: true };
    const doc = makeDoc([rasterLayer('r1', { inpaint: region }), inpaintMask('m1')]);
    const entry = entryOfKind(doc, 'inpaint-mask');
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['m1', 'r1']);
    const ref = entry!.maskLayers![1]!;
    expect(ref.attributeValue).toBe(1);
    expect(ref.sourceRef).toBe('region:image:r1');
    expect(ref.contentSize).toEqual({ height: 48, width: 64 });
  });

  it('unions multiple masks into the denoise-limit entry', () => {
    const doc = makeDoc([inpaintMask('a', { denoiseLimit: 0.3 }), inpaintMask('b')]);
    const entry = entryOfKind(doc, 'inpaint-mask');
    expect(entry!.maskLayers!.map((l) => l.id)).toEqual(['a', 'b']);
  });
});

describe('planComposites — noise-mask entries', () => {
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
  const maskKeyOf = (doc: CanvasDocumentContractV3, kind: string): string | undefined => entryOfKind(doc, kind)?.key;

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
    isHidden: boolean;
    withTransparencyEffect: boolean;
    opacity: number;
    blendMode: CanvasLayerContract['blendMode'];
  }> = {}
): CanvasLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: overrides.blendMode ?? 'normal',
  id,
  isEnabled: overrides.isEnabled ?? true,
  isHidden: overrides.isHidden,
  isLocked: false,
  name: id,
  opacity: overrides.opacity ?? 1,
  source: overrides.source ?? { image: imageRef(id, 64, 48), type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: overrides.withTransparencyEffect ?? false,
});

describe('planComposites — hidden layers still generate', () => {
  it('keeps a hidden control layer in the plan', () => {
    const shown = planControlComposites(makeDoc([controlLayer('c')]), BBOX);
    const hidden = planControlComposites(makeDoc([controlLayer('c', { isHidden: true })]), BBOX);
    expect(hidden.length).toBe(shown.length);
    expect(hidden.length).toBeGreaterThan(0);
  });
});

describe('planComposites — control-layer exclusion', () => {
  it('never includes control layers in the base-raster entry', () => {
    const plan = planComposites(makeDoc([rasterLayer('r'), controlLayer('c')]), BBOX);
    const base = plan.entries.find((entry) => entry.kind === 'base-raster');
    expect(base?.layers.map((layer) => layer.id)).toEqual(['r']);
  });
});

describe('getBaseRasterContentBounds', () => {
  it('returns rounded transformed bounds for enabled raster content only', () => {
    const document = makeDoc([
      rasterLayer('enabled', {
        source: { image: imageRef('source', 100, 50), type: 'image' },
        transform: { rotation: 0, scaleX: 2, scaleY: 1, x: -10.25, y: 20.5 },
      }),
      { ...controlLayer('control'), isEnabled: true },
      { ...rasterLayer('disabled'), isEnabled: false },
    ]);

    expect(getBaseRasterContentBounds(document)).toEqual({ x: -11, y: 20, width: 201, height: 51 });
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
    expect(composites[0]!.entry.key).not.toBe(composites[1]!.entry.key);
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
});

describe('planComposites — raster adjustments in the base key', () => {
  it('folds a non-identity adjustment into the base-raster entry (and its key)', () => {
    const plain = rasterLayer('a');
    const adjusted = rasterLayer('a', {
      adjustments: [
        { brightness: 0.5, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      ],
    });
    const plan = planComposites(makeDoc([adjusted]), BBOX);
    const ref = plan.entries[0]!.layers[0]!;
    expect(ref.adjustments).toEqual(adjusted.type === 'raster' ? adjusted.adjustments : undefined);
    // A changed adjustment changes the entry key so the executor re-composites/uploads.
    expect(keyOf(makeDoc([plain]))).not.toBe(keyOf(makeDoc([adjusted])));
  });
});

describe('planComposites — group gating', () => {
  it('excludes a raster leaf whose group is disabled', () => {
    const doc = {
      ...makeDoc([]),
      stacks: stacksFrom([groupContract('g', [rasterLayer('inner')], { isEnabled: false }), rasterLayer('root')]),
    };
    expect(planComposites(doc, BBOX).entries[0]!.layers.map((layer) => layer.id)).toEqual(['root']);
    expect(getBaseRasterContentBounds(doc)).toEqual({ height: 48, width: 64, x: 0, y: 0 });
  });
});
