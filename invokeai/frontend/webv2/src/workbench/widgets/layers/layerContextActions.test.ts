import type { CanvasDocumentContractV2, CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import { describe, expect, it, vi } from 'vitest';

import {
  getLayerContextActionDefinition,
  getLayerContextActions,
  LAYER_CONTEXT_ACTION_DEFINITIONS,
  type LayerContextAction,
  type LayerContextActionEffects,
  type LayerContextActionId,
  type LayerContextActionRuntimeContext,
  type LayerContextActionState,
} from './layerContextActions';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
} from './layerOps';

const englishCatalogModules = import.meta.glob('../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const en = Object.values(englishCatalogModules)[0] as unknown;

const paintLayer = (id: string, patch: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  ...createEmptyPaintLayer(id, id),
  source: { bitmap: { height: 10, imageName: `${id}.png`, width: 10 }, type: 'paint' },
  ...patch,
});

const rasterLayer = paintLayer('raster');
const nonEmptyControlLayer = {
  ...createControlLayer('Control', 'control'),
  source: { image: { height: 10, imageName: 'control-image', width: 10 }, type: 'image' as const },
};

const makeLayer = (type: CanvasLayerContract['type']): CanvasLayerContract => {
  switch (type) {
    case 'raster':
      return paintLayer('raster');
    case 'control':
      return nonEmptyControlLayer;
    case 'inpaint_mask':
      return createInpaintMaskLayer('Mask', 'mask');
    case 'regional_guidance':
      return createRegionalGuidanceLayer('Region', 0, 'region');
  }
};

const makeDocument = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 512, width: 512, x: 0, y: 0 },
  height: 512,
  layers,
  selectedLayerId: layers[0]?.id ?? null,
  version: 2,
  width: 512,
});

const makeState = (
  layer: CanvasLayerContract,
  overrides: Partial<LayerContextActionState> = {}
): LayerContextActionState => {
  const document = overrides.document ?? makeDocument([layer]);
  return {
    canRunWorkflow: true,
    document,
    hasEngine: true,
    hasSupportedContent: true,
    hasWorkflowBindings: true,
    index: document.layers.findIndex((entry) => entry.id === layer.id),
    interactionLocked: false,
    layer,
    ...overrides,
  };
};

const makeEffects = (): LayerContextActionEffects => ({
  booleanMerge: vi.fn(() => Promise.resolve()),
  copyTo: vi.fn(),
  copyToClipboard: vi.fn(() => Promise.resolve()),
  cropToBbox: vi.fn(() => Promise.resolve()),
  delete: vi.fn(),
  duplicate: vi.fn(),
  extractMaskedArea: vi.fn(() => Promise.resolve()),
  fitToBbox: vi.fn(),
  mergeDown: vi.fn(),
  openProperties: vi.fn(),
  openRename: vi.fn(),
  startWorkflow: vi.fn(),
  startSelectObject: vi.fn(),
  startFilter: vi.fn(),
  patchConfig: vi.fn(),
  rasterize: vi.fn(),
  reorder: vi.fn(),
  saveToAssets: vi.fn(() => Promise.resolve()),
  toggleLock: vi.fn(),
  toggleVisibility: vi.fn(),
  transform: vi.fn(),
  convertTo: vi.fn(),
});

const makeRuntimeContext = (
  layer: CanvasLayerContract,
  overrides: Partial<LayerContextActionRuntimeContext> = {}
): LayerContextActionRuntimeContext => ({
  ...makeState(layer, overrides),
  effects: makeEffects(),
  ...overrides,
});

const byId = (actions: readonly LayerContextAction[], id: LayerContextActionId): LayerContextAction => {
  const action = actions.find((entry) => entry.id === id);
  expect(action, `Expected action ${id}`).toBeDefined();
  return action!;
};

const idsFor = (
  layer: CanvasLayerContract,
  layers: readonly CanvasLayerContract[] = [layer],
  overrides: Partial<LayerContextActionState> = {}
): LayerContextActionId[] =>
  getLayerContextActions(makeState(layer, { document: makeDocument([...layers]), ...overrides })).map(
    (action) => action.id
  );

const getEnglishTranslation = (key: string): unknown =>
  key.split('.').reduce<unknown>((value, segment) => {
    if (!value || typeof value !== 'object') {
      return undefined;
    }
    return (value as Record<string, unknown>)[segment];
  }, en);

describe('layer context action registry', () => {
  it('defines every action as complete executable data', () => {
    for (const definition of LAYER_CONTEXT_ACTION_DEFINITIONS) {
      expect(definition.icon).toBeDefined();
      expect(definition.section).toMatch(/^(quick|primary|operations|output|state|danger)$/);
      expect(definition.order).toEqual(expect.any(Number));
      expect(definition.supportedLayerTypes.length).toBeGreaterThan(0);
      expect(definition.isVisible).toEqual(expect.any(Function));
      expect(definition.isEnabled).toEqual(expect.any(Function));
      expect(definition.handler).toEqual(expect.any(Function));
    }
  });

  it('has one definition per action id', () => {
    const ids = LAYER_CONTEXT_ACTION_DEFINITIONS.map((definition) => definition.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('has English translations for registry labels and action errors', () => {
    const keys = [
      ...LAYER_CONTEXT_ACTION_DEFINITIONS.map((definition) => definition.labelKey),
      'widgets.layers.actions.enableTransparencyEffect',
      'widgets.layers.actions.disableTransparencyEffect',
      'widgets.layers.actions.enableAutoNegative',
      'widgets.layers.actions.disableAutoNegative',
      'widgets.layers.actions.actionFailed',
      'widgets.layers.actions.missing',
      'widgets.layers.actions.unsupported',
      'widgets.layers.actions.busy',
      'widgets.layers.actions.locked',
      'widgets.layers.actions.notReady',
      'widgets.layers.actions.empty',
      'widgets.layers.actions.disabled',
      'widgets.layers.actions.copyFailed',
      'widgets.layers.actions.cropBusy',
      'widgets.layers.actions.cropFailed',
      'widgets.layers.actions.cropUnsupported',
      'widgets.layers.actions.cropped',
      'widgets.layers.rasterFilter.busy',
      'widgets.layers.rasterFilter.cancel',
      'widgets.layers.rasterFilter.commitFailure',
      'widgets.layers.rasterFilter.copy',
      'widgets.layers.rasterFilter.durabilityFailure',
      'widgets.layers.rasterFilter.graphFailure',
      'widgets.layers.rasterFilter.locked',
      'widgets.layers.rasterFilter.preview',
      'widgets.layers.rasterFilter.replace',
      'widgets.layers.rasterFilter.running',
      'widgets.layers.rasterFilter.stale',
      'widgets.layers.rasterFilter.title',
      'widgets.layers.rasterFilter.unsupported',
      'widgets.layers.selectObject.autoProcess',
      'widgets.layers.selectObject.exclude',
      'widgets.layers.selectObject.include',
      'widgets.layers.selectObject.invert',
      'widgets.layers.selectObject.isolatedPreview',
      'widgets.layers.selectObject.model',
      'widgets.layers.selectObject.modelHuge',
      'widgets.layers.selectObject.modelSam2Large',
      'widgets.layers.selectObject.prompt',
      'widgets.layers.selectObject.process',
      'widgets.layers.selectObject.refine',
      'widgets.layers.selectObject.reset',
      'widgets.layers.selectObject.saveAs',
      'widgets.layers.selectObject.saveAs_control',
      'widgets.layers.selectObject.saveAs_inpaint_mask',
      'widgets.layers.selectObject.saveAs_raster',
      'widgets.layers.selectObject.saveAs_regional_guidance',
      'widgets.layers.selectObject.visual',
      'widgets.layers.runWorkflow.aborted',
      'widgets.layers.runWorkflow.busy',
      'widgets.layers.runWorkflow.cancel',
      'widgets.layers.runWorkflow.copyRaster',
      'widgets.layers.runWorkflow.copySuccess',
      'widgets.layers.runWorkflow.destination',
      'widgets.layers.runWorkflow.durabilityFailure',
      'widgets.layers.runWorkflow.disabled',
      'widgets.layers.runWorkflow.empty',
      'widgets.layers.runWorkflow.failed',
      'widgets.layers.runWorkflow.gallery',
      'widgets.layers.runWorkflow.gallerySuccess',
      'widgets.layers.runWorkflow.graphFailure',
      'widgets.layers.runWorkflow.hydrationFailure',
      'widgets.layers.runWorkflow.input',
      'widgets.layers.runWorkflow.locked',
      'widgets.layers.runWorkflow.missing',
      'widgets.layers.runWorkflow.noBindings',
      'widgets.layers.runWorkflow.notReady',
      'widgets.layers.runWorkflow.output',
      'widgets.layers.runWorkflow.replace',
      'widgets.layers.runWorkflow.replaceSuccess',
      'widgets.layers.runWorkflow.run',
      'widgets.layers.runWorkflow.running',
      'widgets.layers.runWorkflow.staging',
      'widgets.layers.runWorkflow.stagingSuccess',
      'widgets.layers.runWorkflow.stale',
      'widgets.layers.runWorkflow.title',
      'widgets.layers.runWorkflow.unsupported',
    ];

    for (const key of keys) {
      expect(getEnglishTranslation(key), key).toEqual(expect.any(String));
    }
  });

  it('provides the disabled-layer reason used when Select Object start is refused', () => {
    expect(getEnglishTranslation('widgets.layers.actions.disabled')).toBe('Enable the layer before using this action.');
  });

  it('throws for an unknown action definition', () => {
    expect(() => getLayerContextActionDefinition('unknown' as LayerContextActionId)).toThrow(
      'Unknown layer context action: unknown'
    );
  });

  it('dispatches a registry handler through injected effects', () => {
    const transform = vi.fn();
    const context = makeRuntimeContext(rasterLayer, { effects: { ...makeEffects(), transform } });
    getLayerContextActionDefinition('transform').handler(context);
    expect(transform).toHaveBeenCalledOnce();
  });

  it('starts Select Object for the action layer through the registry', () => {
    const effects = makeEffects();
    const context = makeRuntimeContext(rasterLayer, { effects });

    getLayerContextActionDefinition('select-object').handler(context);

    expect(effects.startSelectObject).toHaveBeenCalledWith(rasterLayer.id);
    expect(Object.values(effects).filter((effect) => effect.mock.calls.length > 0)).toEqual([
      effects.startSelectObject,
    ]);
  });

  it('starts Run Workflow for the action layer through the registry', () => {
    const effects = makeEffects();
    const context = makeRuntimeContext(rasterLayer, { effects });

    getLayerContextActionDefinition('run-workflow').handler(context);

    expect(effects.startWorkflow).toHaveBeenCalledWith(rasterLayer.id);
    expect(Object.values(effects).filter((effect) => effect.mock.calls.length > 0)).toEqual([effects.startWorkflow]);
  });

  it('dispatches parameterized registry handlers through injected effects', () => {
    const effects = makeEffects();
    const context = makeRuntimeContext(rasterLayer, { effects });

    getLayerContextActionDefinition('move-to-front').handler(context);
    getLayerContextActionDefinition('adjustments').handler(context);
    getLayerContextActionDefinition('filter').handler(context);
    getLayerContextActionDefinition('intersect').handler(context);
    getLayerContextActionDefinition('copy-to-control').handler(context);
    getLayerContextActionDefinition('convert-to-control').handler(context);
    getLayerContextActionDefinition('regional-auto-negative').handler(context);

    expect(effects.reorder).toHaveBeenCalledWith('front', 'move-to-front');
    expect(effects.openProperties).toHaveBeenCalledWith('adjustments');
    expect(effects.startFilter).toHaveBeenCalledWith(rasterLayer.id);
    expect(effects.booleanMerge).toHaveBeenCalledWith('intersect');
    expect(effects.copyTo).toHaveBeenCalledWith('control');
    expect(effects.convertTo).toHaveBeenCalledWith('control');
    expect(effects.patchConfig).toHaveBeenCalledWith('regional-auto-negative');
  });
});

describe('getLayerContextActions', () => {
  it.each([
    ['raster', 'adjustments', true],
    ['control', 'adjustments', false],
    ['inpaint_mask', 'extract-masked-area', true],
    ['control', 'extract-masked-area', false],
  ] as const)('resolves %s visibility for %s', (type, actionId, expected) => {
    const action = getLayerContextActions(makeState(makeLayer(type))).find((item) => item.id === actionId);
    expect(Boolean(action)).toBe(expected);
  });

  it.each([
    ['raster', true],
    ['control', true],
    ['inpaint_mask', false],
    ['regional_guidance', false],
  ] as const)('exposes transform only for engine-supported %s layers', (type, expected) => {
    const action = getLayerContextActions(makeState(makeLayer(type))).find((item) => item.id === 'transform');
    expect(Boolean(action)).toBe(expected);
  });

  it.each([
    ['raster', true],
    ['control', true],
    ['inpaint_mask', false],
    ['regional_guidance', false],
  ] as const)('exposes Select Object only for exportable %s content', (type, expected) => {
    const action = getLayerContextActions(makeState(makeLayer(type))).find((item) => item.id === 'select-object');
    expect(Boolean(action)).toBe(expected);
  });

  it('hides Select Object without supported source content', () => {
    expect(
      getLayerContextActions(makeState(rasterLayer, { hasSupportedContent: false })).some(
        (action) => action.id === 'select-object'
      )
    ).toBe(false);
  });

  it.each([
    ['raster', true],
    ['control', true],
    ['inpaint_mask', false],
    ['regional_guidance', false],
  ] as const)('exposes Run Workflow only for exportable %s content with compatible bindings', (type, expected) => {
    const action = getLayerContextActions(makeState(makeLayer(type))).find((item) => item.id === 'run-workflow');
    expect(Boolean(action)).toBe(expected);
  });

  it('hides Run Workflow when the active workflow has no compatible image bindings', () => {
    expect(
      getLayerContextActions(makeState(rasterLayer, { hasWorkflowBindings: false })).some(
        (action) => action.id === 'run-workflow'
      )
    ).toBe(false);
  });

  it.each([
    ['no runnable binding', { canRunWorkflow: false }],
    ['missing engine', { hasEngine: false }],
    ['locked layer', { layer: { ...rasterLayer, isLocked: true } }],
    ['locked interaction', { interactionLocked: true }],
  ] as const)('disables Run Workflow for %s', (_label, overrides) => {
    const layer = 'layer' in overrides ? overrides.layer : rasterLayer;
    expect(byId(getLayerContextActions(makeState(layer, overrides)), 'run-workflow').isDisabled).toBe(true);
  });

  it.each([
    ['missing engine', { hasEngine: false }],
    ['disabled layer', { layer: { ...rasterLayer, isEnabled: false } }],
    ['locked layer', { layer: { ...rasterLayer, isLocked: true } }],
    ['locked interaction', { interactionLocked: true }],
  ] as const)('disables Select Object for %s', (_label, overrides) => {
    const layer = 'layer' in overrides ? overrides.layer : rasterLayer;
    expect(byId(getLayerContextActions(makeState(layer, overrides)), 'select-object').isDisabled).toBe(true);
  });

  it('disables transform for a hidden layer', () => {
    const hidden = { ...rasterLayer, isEnabled: false };
    expect(byId(getLayerContextActions(makeState(hidden)), 'transform').isDisabled).toBe(true);
  });

  it('disables mutating pixel actions while interaction is locked', () => {
    const rasterActions = getLayerContextActions(makeState(rasterLayer, { interactionLocked: true }));
    const controlActions = getLayerContextActions(makeState(nonEmptyControlLayer, { interactionLocked: true }));
    expect(byId(rasterActions, 'crop-to-bbox').isDisabled).toBe(true);
    expect(byId(rasterActions, 'filter').isDisabled).toBe(true);
    expect(byId(rasterActions, 'transform').isDisabled).toBe(true);
    expect(byId(controlActions, 'filter').isDisabled).toBe(true);
    expect(byId(rasterActions, 'copy-to-clipboard').isDisabled).toBe(true);
  });

  it('disables every layer action while canvas interaction is locked', () => {
    const above = paintLayer('interaction-above');
    const raster = paintLayer('interaction-raster');
    const below = paintLayer('interaction-below');
    const rasterDocument = makeDocument([above, raster, below]);
    const text = paintLayer('interaction-text', {
      source: {
        align: 'left',
        color: '#fff',
        content: 'locked',
        fontFamily: 'Inter',
        fontSize: 32,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      },
    });
    const inpaint = createInpaintMaskLayer('Interaction mask', 'interaction-mask');
    const regional = createRegionalGuidanceLayer('Interaction region', 0, 'interaction-region');
    const contexts = [
      makeState(raster, { document: rasterDocument, interactionLocked: true }),
      makeState(nonEmptyControlLayer, { interactionLocked: true }),
      makeState(inpaint, { interactionLocked: true }),
      makeState(regional, { interactionLocked: true }),
      makeState(text, { interactionLocked: true }),
    ];
    const actions = contexts.flatMap((context) => getLayerContextActions(context));
    const actionsById = new Map(actions.map((action) => [action.id, action]));
    const actionIds: readonly LayerContextActionId[] = [
      'move-to-front',
      'move-forward',
      'move-backward',
      'move-to-back',
      'duplicate',
      'rename',
      'transform',
      'fit-to-bbox',
      'adjustments',
      'save-to-assets',
      'copy-to-clipboard',
      'crop-to-bbox',
      'extract-masked-area',
      'filter',
      'select-object',
      'run-workflow',
      'intersect',
      'cutout',
      'cutaway',
      'exclude',
      'copy-to-raster',
      'copy-to-control',
      'copy-to-inpaint-mask',
      'copy-to-regional-guidance',
      'rasterize',
      'convert-to-control',
      'convert-to-raster',
      'convert-to-inpaint-mask',
      'convert-to-regional-guidance',
      'control-transparency-effect',
      'regional-positive-prompt',
      'regional-negative-prompt',
      'regional-reference-image',
      'regional-auto-negative',
      'inpaint-noise',
      'inpaint-denoise-limit',
      'merge-down',
      'toggle-visibility',
      'toggle-lock',
      'delete',
    ];

    expect(actionIds).toHaveLength(LAYER_CONTEXT_ACTION_DEFINITIONS.length);
    expect(actionIds.filter((id) => !actionsById.has(id))).toEqual([]);
    expect([...new Set(actions.filter((action) => !action.isDisabled).map((action) => action.id))]).toEqual([]);
  });

  it('still allows non-destructive copy and export from a locked layer when interaction is free', () => {
    const actions = getLayerContextActions(makeState({ ...rasterLayer, isLocked: true }));
    expect(byId(actions, 'copy-to-clipboard').isDisabled).toBe(false);
    expect(byId(actions, 'save-to-assets').isDisabled).toBe(false);
  });

  it('disables locked-layer mutations but keeps toggle lock enabled', () => {
    const lockedRaster = { ...rasterLayer, isLocked: true };
    const below = paintLayer('below-locked-raster');
    const actions = getLayerContextActions(makeState(lockedRaster, { document: makeDocument([lockedRaster, below]) }));

    for (const id of [
      'fit-to-bbox',
      'adjustments',
      'filter',
      'select-object',
      'run-workflow',
      'crop-to-bbox',
      'convert-to-control',
      'intersect',
      'merge-down',
      'delete',
    ] as const) {
      expect(byId(actions, id).isDisabled, id).toBe(true);
    }
    expect(byId(actions, 'toggle-lock').isDisabled).toBe(false);

    const lockedControl = { ...nonEmptyControlLayer, isLocked: true };
    const controlActions = getLayerContextActions(makeState(lockedControl));
    expect(byId(controlActions, 'filter').isDisabled).toBe(true);
    expect(byId(controlActions, 'control-transparency-effect').isDisabled).toBe(true);
    expect(byId(controlActions, 'convert-to-raster').isDisabled).toBe(true);

    const lockedText = paintLayer('locked-text', {
      isLocked: true,
      source: {
        align: 'left',
        color: '#fff',
        content: 'locked',
        fontFamily: 'Inter',
        fontSize: 32,
        fontWeight: 400,
        lineHeight: 1.2,
        type: 'text',
      },
    });
    expect(byId(getLayerContextActions(makeState(lockedText)), 'rasterize').isDisabled).toBe(true);
  });

  it('keeps boolean operations visible but disabled when an adjacent raster is locked', () => {
    const upper = { ...paintLayer('locked-upper'), isLocked: true };
    const below = paintLayer('below-locked-upper');
    const actions = getLayerContextActions(makeState(upper, { document: makeDocument([upper, below]) }));

    for (const id of ['intersect', 'cutout', 'cutaway', 'exclude'] as const) {
      expect(byId(actions, id).isDisabled, id).toBe(true);
    }
  });

  it('requires an engine and supported content for pixel actions', () => {
    const noEngine = getLayerContextActions(makeState(rasterLayer, { hasEngine: false }));
    const empty = getLayerContextActions(makeState(rasterLayer, { hasSupportedContent: false }));

    for (const id of ['transform', 'filter', 'save-to-assets', 'copy-to-clipboard', 'crop-to-bbox'] as const) {
      expect(byId(noEngine, id).isDisabled, id).toBe(true);
    }
    for (const id of ['save-to-assets', 'copy-to-clipboard', 'crop-to-bbox'] as const) {
      expect(byId(empty, id).isDisabled, id).toBe(true);
    }
  });

  it('does not offer fit for cache-only content whose bounds are not yet persisted', () => {
    const cacheOnly = createEmptyPaintLayer('Cache only', 'cache-only');
    const actions = getLayerContextActions(makeState(cacheOnly, { hasSupportedContent: true }));

    expect(byId(actions, 'save-to-assets').isDisabled).toBe(false);
    expect(byId(actions, 'fit-to-bbox').isDisabled).toBe(true);
  });

  it('keeps common actions available for a raster layer', () => {
    expect(idsFor(rasterLayer)).toEqual(
      expect.arrayContaining([
        'move-to-front',
        'duplicate',
        'rename',
        'transform',
        'fit-to-bbox',
        'adjustments',
        'save-to-assets',
        'copy-to-clipboard',
        'crop-to-bbox',
        'merge-down',
        'delete',
      ])
    );
  });

  it('resolves movement boundaries within the layer type group', () => {
    const first = paintLayer('first');
    const middle = paintLayer('middle');
    const last = paintLayer('last');
    const document = makeDocument([first, middle, last]);

    const firstActions = getLayerContextActions(makeState(first, { document }));
    expect(byId(firstActions, 'move-to-front').isDisabled).toBe(true);
    expect(byId(firstActions, 'move-forward').isDisabled).toBe(true);
    expect(byId(firstActions, 'move-backward').isDisabled).toBe(false);
    expect(byId(firstActions, 'move-to-back').isDisabled).toBe(false);

    const middleActions = getLayerContextActions(makeState(middle, { document }));
    expect(byId(middleActions, 'move-forward').isDisabled).toBe(false);
    expect(byId(middleActions, 'move-backward').isDisabled).toBe(false);

    const lastActions = getLayerContextActions(makeState(last, { document }));
    expect(byId(lastActions, 'move-forward').isDisabled).toBe(false);
    expect(byId(lastActions, 'move-backward').isDisabled).toBe(true);
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

    expect(idsFor(paint)).toEqual(
      expect.arrayContaining([
        'copy-to-control',
        'copy-to-inpaint-mask',
        'copy-to-regional-guidance',
        'convert-to-control',
        'convert-to-inpaint-mask',
        'convert-to-regional-guidance',
      ])
    );
    expect(idsFor(text)).not.toContain('convert-to-control');
    expect(idsFor(text)).not.toContain('copy-to-inpaint-mask');
  });

  it('exposes control-only transparency and raster conversion actions while filtering supported control content', () => {
    const empty = createControlLayer('Control', 'control-empty');

    expect(idsFor(empty, [empty], { hasSupportedContent: false })).toEqual(
      expect.arrayContaining([
        'control-transparency-effect',
        'convert-to-raster',
        'copy-to-raster',
        'copy-to-inpaint-mask',
        'copy-to-regional-guidance',
      ])
    );
    expect(idsFor(empty, [empty], { hasSupportedContent: false })).not.toContain('filter');
    expect(idsFor(empty, [empty], { hasSupportedContent: true })).toContain('filter');
    expect(idsFor(nonEmptyControlLayer)).toContain('filter');
  });

  it('exposes filter for every supported raster source', () => {
    const layers: CanvasRasterLayerContractV2[] = [
      paintLayer('filter-image', {
        source: { image: { height: 10, imageName: 'filter-image', width: 10 }, type: 'image' },
      }),
      paintLayer('filter-paint'),
      paintLayer('filter-text', {
        source: {
          align: 'left',
          color: '#fff',
          content: 'Filter me',
          fontFamily: 'Inter',
          fontSize: 32,
          fontWeight: 400,
          lineHeight: 1.2,
          type: 'text',
        },
      }),
      paintLayer('filter-gradient', {
        source: {
          angle: 45,
          height: 20,
          kind: 'linear',
          stops: [
            { color: '#000', offset: 0 },
            { color: '#fff', offset: 1 },
          ],
          type: 'gradient',
          width: 20,
        },
      }),
      paintLayer('filter-rect', {
        source: { fill: '#fff', height: 20, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 20 },
      }),
      paintLayer('filter-ellipse', {
        source: {
          fill: '#fff',
          height: 20,
          kind: 'ellipse',
          stroke: null,
          strokeWidth: 0,
          type: 'shape',
          width: 20,
        },
      }),
    ];

    for (const layer of layers) {
      expect(idsFor(layer), layer.id).toContain('filter');
    }
  });

  it('hides filter for empty paint and polygon raster sources', () => {
    const empty = createEmptyPaintLayer('Empty', 'filter-empty');
    const polygon = paintLayer('filter-polygon', {
      source: {
        fill: '#fff',
        height: 20,
        kind: 'polygon',
        points: [
          { x: 0, y: 0 },
          { x: 20, y: 0 },
          { x: 10, y: 20 },
        ],
        stroke: null,
        strokeWidth: 0,
        type: 'shape',
        width: 20,
      },
    });

    expect(idsFor(empty, [empty], { hasSupportedContent: false })).not.toContain('filter');
    expect(idsFor(polygon, [polygon], { hasSupportedContent: true })).not.toContain('filter');
  });

  it('enables raster filter only with an engine, unlocked layer, and unlocked interaction', () => {
    const unlocked = paintLayer('filter-enabled');
    expect(byId(getLayerContextActions(makeState(unlocked)), 'filter').isDisabled).toBe(false);
    expect(byId(getLayerContextActions(makeState(unlocked, { hasEngine: false })), 'filter').isDisabled).toBe(true);
    expect(byId(getLayerContextActions(makeState({ ...unlocked, isLocked: true })), 'filter').isDisabled).toBe(true);
    expect(byId(getLayerContextActions(makeState(unlocked, { interactionLocked: true })), 'filter').isDisabled).toBe(
      true
    );
  });

  it('does not expose copy-to-raster for raster layers', () => {
    expect(idsFor(rasterLayer)).not.toContain('copy-to-raster');
  });

  it('exposes regional guidance add actions only for missing prompts', () => {
    const layer = { ...createRegionalGuidanceLayer('Region', 0, 'region-prompts'), positivePrompt: 'already' };

    expect(idsFor(layer)).toContain('regional-negative-prompt');
    expect(idsFor(layer)).toContain('regional-reference-image');
    expect(idsFor(layer)).toContain('regional-auto-negative');
    expect(idsFor(layer)).not.toContain('regional-positive-prompt');
  });

  it('exposes inpaint modifier add actions only for missing modifiers', () => {
    const layer = { ...createInpaintMaskLayer('Mask', 'mask-modifiers'), noiseLevel: 0.25 };

    expect(idsFor(layer)).toContain('inpaint-denoise-limit');
    expect(idsFor(layer)).toContain('copy-to-regional-guidance');
    expect(idsFor(layer)).toContain('extract-masked-area');
    expect(idsFor(layer)).not.toContain('inpaint-noise');
  });

  it('copies regional guidance to an inpaint mask without unsupported conversions', () => {
    const layer = createRegionalGuidanceLayer('Region', 0, 'region-copy');

    expect(idsFor(layer)).toContain('copy-to-inpaint-mask');
    expect(idsFor(layer)).not.toContain('convert-to-inpaint-mask');
    expect(idsFor(layer)).not.toContain('copy-to-control');
  });

  it('exposes boolean raster actions only for an adjacent mergeable raster pair', () => {
    const upper = paintLayer('upper');
    const below = paintLayer('below');
    const operations = ['intersect', 'cutout', 'cutaway', 'exclude'];

    expect(idsFor(upper, [upper, below])).toEqual(expect.arrayContaining(operations));
    expect(idsFor(below, [upper, below])).toEqual(expect.not.arrayContaining(operations));
    expect(idsFor(upper, [upper, createInpaintMaskLayer('Mask', 'mask-below')])).toEqual(
      expect.not.arrayContaining(operations)
    );
  });
});
