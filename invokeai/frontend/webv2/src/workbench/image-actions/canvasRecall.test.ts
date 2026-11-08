import type * as GalleryModule from '@features/gallery';
import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { compileCanvasGraph } from '@features/generation/canvasGraph';
import { setArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient } from '@tanstack/react-query';
import * as CanvasOperations from '@workbench/canvas-operations/api';
import { recordCanvasMetadata } from '@workbench/canvas-operations/saveCanvasToGallery';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ metadata: vi.fn(), resolveMany: vi.fn(), makeCanvasAsset: vi.fn() }));
vi.mock('@features/gallery', async (original) => ({
  ...(await original<typeof GalleryModule>()),
  galleryImages: api,
  galleryDurability: { makeCanvasAsset: api.makeCanvasAsset },
}));
vi.mock('@features/gallery/data/backend', async (original) => ({
  ...(await original<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: api.metadata,
}));

import { commitCanvasRecall, prepareCanvasRecall } from './canvasRecall';
import { executeImageRecall } from './executeImageRecall';
import { executeRecallParameters } from './executeRecallParameters';

const main: MainModelConfig = { base: 'sdxl', key: 'main', name: 'SDXL', type: 'main' };
const controlModel = {
  base: 'sdxl',
  key: 'control-installed',
  name: 'Edges',
  type: 'controlnet',
  hash: 'control-hash',
} as ModelConfig;
const models = [main as ModelConfig, controlModel];
const t = ((key: string) => key) as TFunction;
const base = {
  id: 'layer',
  name: 'Authored layer',
  isEnabled: true,
  isLocked: false,
  opacity: 0.7,
  blendMode: 'normal' as const,
  transform: { x: -40, y: 70, scaleX: 0.5, scaleY: -1, rotation: 0.2 },
};
const source = { type: 'image' as const, image: { imageName: 'map.png', width: 640, height: 480 } };
const control = (): CanvasControlLayerContract => ({
  ...base,
  id: 'control',
  type: 'control',
  source,
  withTransparencyEffect: true,
  adapter: { kind: 'controlnet', model: null, weight: 0.5, beginEndStepPct: [0.2, 0.8], controlMode: 'more_control' },
});
const document = () => ({
  ...createEmptyCanvasDocument(2048, 1536),
  bbox: { x: -128, y: 96, width: 768, height: 1024 },
  stacks: {
    raster: [{ ...base, id: 'raster', type: 'raster', source } as CanvasLayerContract],
    control: [control()],
    inpaint_mask: [
      {
        ...base,
        id: 'mask',
        type: 'inpaint_mask',
        mask: {
          bitmap: { imageName: 'mask.png', width: 320, height: 200 },
          offset: { x: -20, y: 40 },
          fill: { style: 'solid', color: 'red' },
        },
        noise: { level: 0.4, isEnabled: true },
      } as CanvasLayerContract,
    ],
    regional_guidance: [
      {
        ...base,
        id: 'region',
        type: 'regional_guidance',
        positivePrompt: 'A bird',
        negativePrompt: 'blue',
        autoNegative: true,
        mask: { bitmap: null, fill: { style: 'grid', color: 'red' } },
        referenceImages: [],
      } as CanvasLayerContract,
    ],
  },
});
const record = () =>
  recordCanvasMetadata(document(), {
    controls: [
      {
        id: 'control',
        imageName: 'executed-composite.png',
        model: controlModel,
        kind: 'controlnet',
        weight: 0.5,
        beginEndStepPct: [0.2, 0.8],
        controlMode: 'more_control',
      },
    ],
    settings: { strength: 0.35 },
  });
const largeRecord = () => {
  const document = createEmptyCanvasDocument();
  document.stacks.raster = Array.from({ length: 2006 }, (_, index) => ({
    ...base,
    id: `layer-${index}`,
    name: `Layer ${index}`,
    type: 'raster',
    source: {
      type: 'image',
      image: { imageName: `asset-${index === 2005 ? 0 : index}.png`, width: 64, height: 64 },
    },
  }));
  return recordCanvasMetadata(document, { settings: { strength: 0.35 } });
};
const targetFor = (store: ReturnType<typeof createWorkbenchStore>) => {
  const projectId = store.queries.getSnapshot().activeProject.id;
  return { commands: store.commands, projectId, getProject: () => store.queries.getProject(projectId) ?? null };
};

seedArchitectureCapabilities();
beforeEach(() => {
  accountLifecycle.activate('canvas-recall-tests');
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  api.metadata.mockReset();
  api.makeCanvasAsset.mockReset().mockResolvedValue(undefined);
  api.resolveMany.mockImplementation((names: string[]) => Promise.resolve(names.map((imageName) => ({ imageName }))));
});

afterEach(() => vi.restoreAllMocks());

describe('canvas record to recall', () => {
  it('recalls a compiled output with executed automatic controls, transformed sources, masks, regions and authored geometry', async () => {
    const authored = document();
    const canvasRecord = record();
    const compiled = compileCanvasGraph({
      model: main,
      settings: { ...getDefaultGenerateSettings(main), seed: 12 },
      projectSettings: { useCpuNoise: true, useNsfwChecker: false, useWatermark: false },
      mode: 'txt2img',
      destination: 'gallery',
      bbox: authored.bbox,
      compositeImageName: null,
      compositing: DEFAULT_CANVAS_COMPOSITING,
      strength: 0.35,
      canvasMetadata: canvasRecord,
      controlLayers: [
        {
          id: 'control',
          imageName: 'executed-composite.png',
          model: controlModel,
          kind: 'controlnet',
          weight: 0.5,
          beginEndStepPct: [0.2, 0.8],
          controlMode: 'more_control',
        },
      ],
    });
    const metadata = Object.values(compiled.backendGraph.nodes).find((node) => node.type === 'core_metadata')!;
    expect(metadata.control_layers).toEqual([
      {
        type: 'controlnet',
        layer_id: 'control',
        model: controlModel,
        image: { image_name: 'executed-composite.png', width: 768, height: 1024 },
        position: { x: -128, y: 96 },
        weight: 0.5,
        begin_step_percent: 0.2,
        end_step_percent: 0.8,
        control_mode: 'more_control',
      },
    ]);
    expect(metadata.controlnets).toEqual([
      {
        image: { image_name: 'executed-composite.png', width: 768, height: 1024 },
        control_model: controlModel,
        control_weight: 0.5,
        begin_step_percent: 0.2,
        end_step_percent: 0.8,
        control_mode: 'more_control',
      },
    ]);
    // Later UI mutations cannot rewrite the captured record.
    authored.stacks.control[0]!.adapter.weight = 1.5;
    const store = createWorkbenchStore();
    const target = targetFor(store);
    const previous = target.getProject()!.canvas.document;
    const plan = await prepareCanvasRecall({ metadata, model: main, models, owner: captureAccountScope() });
    expect(plan.problems).toEqual([]);
    expect(commitCanvasRecall({ plan, target, expectedDocument: previous, restorePointName: 'Before recall' })).toBe(
      true
    );
    const recalled = target.getProject()!;
    expect(recalled.canvas.document.bbox).toEqual({ x: -128, y: 96, width: 768, height: 1024 });
    expect(recalled.canvas.document.stacks.control[0]).toMatchObject({
      source,
      transform: base.transform,
      adapter: { model: 'control-installed', weight: 0.5 },
    });
    expect(recalled.canvas.document.stacks.inpaint_mask[0]).toMatchObject({
      mask: { offset: { x: -20, y: 40 } },
      noise: { level: 0.4 },
    });
    expect(recalled.canvas.document.stacks.regional_guidance[0]).toMatchObject({
      positivePrompt: 'A bird',
      autoNegative: true,
    });
    expect(recalled.canvas.snapshots.at(-1)?.document).toEqual(previous);
    expect(getProjectWidgetValues(recalled, 'canvas').denoisingStrength).toBe(0.35);
    expect(api.resolveMany).toHaveBeenCalledOnce();
    expect(api.resolveMany.mock.calls[0]![0].sort()).toEqual(['map.png', 'mask.png']);
  });

  it.each([false, true])('resolves a large record in bounded batches, with missing assets: %s', async (missing) => {
    const missingNames = missing ? new Set(['asset-1000.png', 'asset-2004.png']) : new Set<string>();
    api.resolveMany.mockImplementation((names: string[]) => {
      if (names.length > 1000) {
        return Promise.reject(new Error('422: too many image names'));
      }
      return Promise.resolve(names.filter((name) => !missingNames.has(name)).map((imageName) => ({ imageName })));
    });
    const plan = await prepareCanvasRecall({
      metadata: { canvas_v3_metadata: largeRecord() },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    expect(api.resolveMany.mock.calls.map(([names]) => names.length)).toEqual([1000, 1000, 5]);
    const requestedNames = api.resolveMany.mock.calls.flatMap(([names]) => names);
    expect(requestedNames).toHaveLength(2005);
    expect(new Set(requestedNames).size).toBe(2005);
    expect(plan.document?.stacks.raster).toHaveLength(missing ? 2004 : 2006);
    expect(plan.problems).toEqual(missing ? ['Layer 1000', 'Layer 2004'] : []);
    expect(plan.partial).toBe(missing);
  });

  it.each(['account', 'project', 'document'] as const)(
    'stops large-record asset reads when the originating %s changes between batches',
    async (change) => {
      const store = createWorkbenchStore();
      const target = targetFor(store);
      const originalDocument = target.getProject()!.canvas.document;
      let release!: (images: { imageName: string }[]) => void;
      api.resolveMany.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            release = resolve;
          })
      );
      const owner = captureAccountScope();
      const pending = prepareCanvasRecall({
        metadata: { canvas_v3_metadata: largeRecord() },
        model: main,
        models,
        owner,
        isCurrent: () =>
          store.queries.isActiveProject(target.projectId) && target.getProject()?.canvas.document === originalDocument,
      });
      expect(api.resolveMany).toHaveBeenCalledOnce();
      if (change === 'account') {
        accountLifecycle.activate('another-account');
      } else if (change === 'project') {
        store.commands.projects.create();
      } else {
        store.commands.canvas.apply(target.projectId, {
          type: 'setCanvasBbox',
          bbox: { ...originalDocument.bbox, x: 100 },
        });
      }
      release(Array.from({ length: 1000 }, (_, index) => ({ imageName: `asset-${index}.png` })));
      if (change === 'account') {
        await expect(pending).rejects.toThrow();
      } else {
        const plan = await pending;
        expect(plan.document).toBeNull();
        expect(plan.settings).toEqual({});
        expect(plan.problems).toEqual(['canvas-changed']);
      }
      expect(api.resolveMany).toHaveBeenCalledOnce();
    }
  );

  it('resolves a reinstalled portable control identity and refuses incompatible, ambiguous and missing assets without clearing work', async () => {
    const saved = record();
    const reinstalled = { ...controlModel, key: 'reinstalled' };
    const restored = await prepareCanvasRecall({
      metadata: { canvas_v3_metadata: saved },
      model: main,
      models: [reinstalled],
      owner: captureAccountScope(),
    });
    expect(restored.document?.stacks.control[0]).toMatchObject({ adapter: { model: 'reinstalled' } });
    api.resolveMany.mockResolvedValue([{ imageName: 'mask.png' }]);
    const partial = await prepareCanvasRecall({
      metadata: { canvas_v3_metadata: saved },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    const store = createWorkbenchStore(),
      target = targetFor(store),
      original = target.getProject()!.canvas.document;
    const originalIds = original.stacks.inpaint_mask.map((node) => node.id);
    expect(partial.partial).toBe(true);
    commitCanvasRecall({ plan: partial, target, expectedDocument: original, restorePointName: 'Before recall' });
    expect(target.getProject()!.canvas.document.stacks.inpaint_mask.map((node) => node.id)).toEqual(
      expect.arrayContaining(originalIds)
    );
    expect(target.getProject()!.canvas.snapshots).toHaveLength(0);
    api.resolveMany.mockImplementation((names: string[]) => Promise.resolve(names.map((imageName) => ({ imageName }))));
    for (const catalog of [
      [{ ...controlModel, base: 'sd-1' }],
      [],
      [reinstalled, { ...reinstalled, key: 'duplicate' }],
    ]) {
      const result = await prepareCanvasRecall({
        metadata: { canvas_v3_metadata: saved },
        model: main,
        models: catalog,
        owner: captureAccountScope(),
      });
      expect(result.document?.stacks.control).toEqual([]);
      expect(result.partial).toBe(true);
    }
  });

  it('distinguishes absence, explicit empty, malformed and future records; fences changes and project closure', async () => {
    const store = createWorkbenchStore(),
      target = targetFor(store),
      before = target.getProject()!.canvas.document;
    const prepare = (metadata: unknown) =>
      prepareCanvasRecall({ metadata, model: main, models, owner: captureAccountScope() });
    expect((await prepare({})).document).toBeNull();
    for (const raw of [null, { version: 2 }, { version: 1, document: { version: 3, stacks: { control: 'wrong' } } }]) {
      const plan = await prepare({ canvas_v3_metadata: raw });
      expect(plan.document).toBeNull();
      expect(plan.problems).not.toEqual([]);
    }
    const empty = await prepare({ canvas_v3_metadata: recordCanvasMetadata(createEmptyCanvasDocument()) });
    expect(empty.document).not.toBeNull();
    expect(
      commitCanvasRecall({
        plan: empty,
        target: { ...target, getProject: () => null },
        expectedDocument: before,
        restorePointName: 'Before recall',
      })
    ).toBe(false);
    store.commands.canvas.apply(target.projectId, { type: 'setCanvasBbox', bbox: { ...before.bbox, x: 100 } });
    expect(
      commitCanvasRecall({ plan: empty, target, expectedDocument: before, restorePointName: 'Before recall' })
    ).toBe(false);
    expect(target.getProject()!.canvas.document.bbox.x).toBe(100);
    const latest = target.getProject()!.canvas.document;
    expect(
      commitCanvasRecall({
        plan: await prepare({ canvas_v3_metadata: recordCanvasMetadata(createEmptyCanvasDocument()) }),
        target,
        expectedDocument: latest,
        restorePointName: 'Before recall',
      })
    ).toBe(true);
    expect(target.getProject()!.canvas.document.stacks.inpaint_mask).toEqual([]);
  });

  it.each(['all', 'remix'] as const)(
    'restores canvas through the shared image %s action used by gallery, preview and staging',
    async (kind) => {
      const store = createWorkbenchStore(),
        target = targetFor(store);
      api.metadata.mockResolvedValue({ canvas_v3_metadata: record() });
      const image = { imageName: 'output.png', width: 768, height: 1024 } as GalleryImage;
      expect(
        await executeImageRecall({
          commands: store.commands,
          canvasTarget: target,
          getGenerateValues: () => ({ modelKey: main.key }),
          image,
          kind,
          models,
          projectId: target.projectId,
          queryClient: new QueryClient(),
          t,
        })
      ).toBe(true);
      expect(target.getProject()!.canvas.document.stacks.control[0]).toMatchObject({
        adapter: { model: controlModel.key },
      });
    }
  );

  it.each(['all', 'remix', 'api'] as const)(
    'preserves the whole Canvas and recalls valid Generate fields when the %s asset lookup fails',
    async (surface) => {
      const store = createWorkbenchStore();
      const target = targetFor(store);
      const before = target.getProject()!;
      const canvasSettings = getProjectWidgetValues(before, 'canvas');
      const notifications = vi.spyOn(store.commands.notifications, 'add');
      const feedbackT = ((key: string, values?: Record<string, string>) =>
        key.endsWith('image-lookup-failed')
          ? 'Canvas images could not be checked'
          : (values?.items ?? values?.detail ?? key)) as TFunction;
      const parameters = {
        positive_prompt: 'A recalled landscape',
        seed: 456,
        steps: 31,
        denoise_strength: 0.4,
        control_layers: [{ model_key: controlModel.key, image: { image_name: 'map.png', width: 640, height: 480 } }],
      };
      api.metadata.mockResolvedValue({ ...parameters, canvas_v3_metadata: largeRecord() });
      if (surface === 'api') {
        api.resolveMany.mockRejectedValueOnce(new TypeError('Failed to fetch'));
      } else {
        // The second batch fails after some assets were verified; none of the Canvas may replace authored work.
        api.resolveMany.mockImplementationOnce((names: string[]) =>
          Promise.resolve(names.map((imageName) => ({ imageName })))
        );
        api.resolveMany.mockRejectedValueOnce(new ApiError('Image service unavailable', 503));
      }
      const applied =
        surface === 'api'
          ? await executeRecallParameters({
              commands: store.commands,
              canvasTarget: target,
              getGenerateValues: () => ({ modelKey: main.key }),
              models,
              owner: captureAccountScope(),
              parameters,
              projectId: target.projectId,
              t: feedbackT,
            })
          : await executeImageRecall({
              commands: store.commands,
              canvasTarget: target,
              getGenerateValues: () => ({ modelKey: main.key }),
              image: { imageName: 'out.png', width: 512, height: 512 } as GalleryImage,
              kind: surface,
              models,
              projectId: target.projectId,
              queryClient: new QueryClient(),
              t: feedbackT,
            });
      expect(applied).toBe(true);
      const after = target.getProject()!;
      expect(after.canvas).toBe(before.canvas);
      expect(getProjectWidgetValues(after, 'canvas')).toEqual(canvasSettings);
      expect(getProjectWidgetValues(after, 'generate')).toMatchObject({
        positivePrompt: 'A recalled landscape',
        steps: 31,
      });
      if (surface !== 'remix') {
        expect(getProjectWidgetValues(after, 'generate').seed).toBe(456);
      }
      expect(notifications).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'info', message: 'Canvas images could not be checked' })
      );
      expect(api.resolveMany).toHaveBeenCalledTimes(surface === 'api' ? 1 : 2);
    }
  );

  it.each([
    ['gallery', 'account'],
    ['api', 'account'],
    ['gallery', 'project'],
    ['api', 'project'],
  ] as const)(
    'keeps the %s caller inert when its originating %s changes before lookup rejects',
    async (surface, change) => {
      const store = createWorkbenchStore();
      const target = targetFor(store);
      const owner = captureAccountScope();
      let active = true;
      const canvasTarget = { ...target, getProject: () => (active ? target.getProject() : null) };
      const notifications = vi.spyOn(store.commands.notifications, 'add');
      const reportError = vi.spyOn(store.commands.notifications, 'reportError');
      const settings = vi.spyOn(store.commands.generation, 'setSettings');
      let reject!: (error: unknown) => void;
      api.resolveMany.mockImplementationOnce(
        () =>
          new Promise((_, rejectLookup) => {
            reject = rejectLookup;
          })
      );
      const parameters = {
        positive_prompt: 'Do not recall',
        denoise_strength: 0.4,
        control_layers: [{ model_key: controlModel.key, image: { image_name: 'map.png', width: 640, height: 480 } }],
      };
      api.metadata.mockResolvedValue({ ...parameters, canvas_v3_metadata: record() });
      const pending =
        surface === 'api'
          ? executeRecallParameters({
              commands: store.commands,
              canvasTarget,
              getGenerateValues: () => ({ modelKey: main.key }),
              models,
              owner,
              parameters,
              projectId: target.projectId,
              t,
            })
          : executeImageRecall({
              commands: store.commands,
              canvasTarget,
              getGenerateValues: () => ({ modelKey: main.key }),
              image: { imageName: 'out.png', width: 512, height: 512 } as GalleryImage,
              kind: 'all',
              models,
              owner,
              projectId: target.projectId,
              queryClient: new QueryClient(),
              t,
            });
      await vi.waitFor(() => expect(api.resolveMany).toHaveBeenCalled());
      const before = target.getProject();
      if (change === 'account') {
        accountLifecycle.activate('another-account');
      } else {
        active = false;
      }
      reject(new ApiError('Image service unavailable', 503));
      expect(await pending).toBe(false);
      expect(target.getProject()).toBe(before);
      expect(settings).not.toHaveBeenCalled();
      expect(notifications).not.toHaveBeenCalled();
      expect(reportError).not.toHaveBeenCalled();
    }
  );

  it('does not classify asset-mapping programmer errors as unavailable images', async () => {
    const failure = new TypeError('Cannot read properties of undefined');
    api.resolveMany.mockRejectedValueOnce(failure);
    await expect(
      prepareCanvasRecall({
        metadata: { canvas_v3_metadata: record() },
        model: main,
        models,
        owner: captureAccountScope(),
      })
    ).rejects.toBe(failure);
  });

  it('recalls API controls using processed maps and denoise strength, preserving other stacks, then appends or explicitly clears only controls', async () => {
    const store = createWorkbenchStore(),
      target = targetFor(store),
      originalMask = target.getProject()!.canvas.document.stacks.inpaint_mask;
    const execute = (parameters: Record<string, unknown>) =>
      executeRecallParameters({
        commands: store.commands,
        canvasTarget: target,
        getGenerateValues: () => ({ modelKey: main.key }),
        models,
        owner: captureAccountScope(),
        parameters,
        projectId: target.projectId,
        t,
      });
    const control = {
      model_key: controlModel.key,
      image: { image_name: 'original.png', width: 640, height: 480 },
      processed_image: { image_name: 'map.png', width: 640, height: 480 },
      weight: 0.6,
    };
    expect(await execute({ control_layers: [control], denoise_strength: 0.45 })).toBe(true);
    expect(target.getProject()!.canvas.document.stacks.control[0]).toMatchObject({ source, adapter: { weight: 0.6 } });
    expect(target.getProject()!.canvas.document.stacks.inpaint_mask).toEqual(originalMask);
    expect(await execute({ append: true, control_layers: [control] })).toBe(true);
    expect(target.getProject()!.canvas.document.stacks.control).toHaveLength(2);
    expect(await execute({ control_layers: [] })).toBe(true);
    expect(target.getProject()!.canvas.document.stacks.control).toEqual([]);
    expect(target.getProject()!.canvas.document.stacks.inpaint_mask).toEqual(originalMask);
    expect(await execute({ control_layers: [control] })).toBe(true);
    expect(await execute({ control_layers: null, denoise_strength: null })).toBe(true);
    expect(target.getProject()!.canvas.document.stacks.control).toEqual([]);
    expect(getProjectWidgetValues(target.getProject()!, 'canvas').denoisingStrength).toBe(0.75);
    const beforeMalformed = target.getProject()!.canvas.document;
    expect(await execute({ control_layers: [{ ...control, weight: 'not a weight' }] })).toBe(false);
    expect(target.getProject()!.canvas.document).toBe(beforeMalformed);
  });

  it('reads legacy image-backed controls, rasters and masks without treating the legacy empty record as Clear Canvas', async () => {
    const legacyBase = {
      id: 'old',
      name: 'Old layer',
      isEnabled: true,
      isLocked: false,
      position: { x: -10, y: 20 },
      opacity: 1,
      objects: [{ id: 'image', type: 'image', image: { image_name: 'map.png', width: 640, height: 480 } }],
    };
    const legacy = {
      rasterLayers: [{ ...legacyBase, id: 'raster' }],
      controlLayers: [
        {
          ...legacyBase,
          id: 'control',
          controlAdapter: {
            type: 'controlnet',
            model: controlModel,
            weight: 0.75,
            controlMode: 'balanced',
            beginEndStepPct: [0, 1],
          },
        },
      ],
      inpaintMasks: [{ ...legacyBase, id: 'mask', fill: { style: 'grid', color: { r: 255, g: 0, b: 0 } } }],
      regionalGuidance: [],
    };
    const plan = await prepareCanvasRecall({
      metadata: { canvas_v2_metadata: legacy },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    expect(plan.problems).toEqual([]);
    expect(plan.document?.stacks.control[0]).toMatchObject({
      transform: { x: -10, y: 20 },
      adapter: { model: controlModel.key },
    });
    expect(plan.document?.stacks.inpaint_mask[0]).toMatchObject({ mask: { bitmap: { imageName: 'map.png' } } });
    const empty = await prepareCanvasRecall({
      metadata: { canvas_v2_metadata: { rasterLayers: [], controlLayers: [], inpaintMasks: [], regionalGuidance: [] } },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    expect(empty.document).toBeNull();
  });

  it('rejects account-stale lookups before commit', async () => {
    let release!: (value: unknown[]) => void;
    api.resolveMany.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    );
    const pending = prepareCanvasRecall({
      metadata: { canvas_v3_metadata: record() },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    accountLifecycle.activate('another-account');
    release([{ imageName: 'map.png' }, { imageName: 'mask.png' }]);
    await expect(pending).rejects.toThrow();
  });
});

it('keeps a staged generation session while adding recalled layers and refuses a stale project after asset lookup', async () => {
  const store = createWorkbenchStore(),
    target = targetFor(store);
  const project = target.getProject()!;
  store.commands.canvas.appendStagingCandidate({
    projectId: target.projectId,
    candidate: {
      imageName: 'staged.png',
      imageUrl: '',
      thumbnailUrl: '',
      height: 100,
      width: 100,
      queuedAt: '',
      sourceQueueItemId: '',
      placement: { x: 0, y: 0, width: 100, height: 100, opacity: 1 },
    },
  });
  const before = target.getProject()!;
  const plan = await prepareCanvasRecall({
    metadata: { canvas_v3_metadata: record() },
    model: main,
    models,
    owner: captureAccountScope(),
  });
  commitCanvasRecall({ plan, target, expectedDocument: before.canvas.document, restorePointName: 'Before recall' });
  expect(target.getProject()!.canvas.stagingArea).toEqual(before.canvas.stagingArea);
  expect(target.getProject()!.canvas.documentRevision).toBe(project.canvas.documentRevision);
  expect(plan.problems).toContain('canvas-pending-results');
  expect(target.getProject()!.canvas.document.stacks.control).toHaveLength(1);

  let release!: (value: unknown[]) => void;
  api.resolveMany.mockImplementation(
    () =>
      new Promise((resolve) => {
        release = resolve;
      })
  );
  api.metadata.mockResolvedValue({
    positive_prompt: 'do not apply to a different project',
    canvas_v3_metadata: record(),
  });
  let active = true;
  api.resolveMany.mockClear();
  const pending = executeImageRecall({
    commands: store.commands,
    canvasTarget: { ...target, getProject: () => (active ? target.getProject() : null) },
    getGenerateValues: () => ({ modelKey: main.key }),
    image: { imageName: 'out.png' } as GalleryImage,
    kind: 'all',
    models,
    projectId: target.projectId,
    queryClient: new QueryClient(),
    t,
  });
  await vi.waitFor(() => expect(api.resolveMany).toHaveBeenCalled());
  const current = target.getProject();
  active = false;
  release([{ imageName: 'map.png' }, { imageName: 'mask.png' }]);
  expect(await pending).toBe(false);
  expect(target.getProject()).toBe(current);
});

it('recalls complete Canvas processing settings through their owner bounds and skips damaged control values', async () => {
  const saved = recordCanvasMetadata(document(), {
    models: [controlModel],
    settings: {
      strength: 0.6,
      compositing: {
        ...DEFAULT_CANVAS_COMPOSITING,
        infillTileSize: 64,
        infillPatchmatchDownscaleSize: 10,
        maskBlur: 48,
        coherenceMode: 'Staged',
      },
      scaling: { method: 'manual', width: 1536, height: 768 },
    },
  });
  const plan = await prepareCanvasRecall({
    metadata: { canvas_v3_metadata: saved },
    model: main,
    models,
    owner: captureAccountScope(),
  });
  expect(plan.settings).toMatchObject({
    denoisingStrength: 0.6,
    infillTileSize: 64,
    infillPatchmatchDownscaleSize: 10,
    maskBlur: 48,
    coherenceMode: 'Staged',
    scaleMethod: 'manual',
    scaledWidth: 1536,
    scaledHeight: 768,
  });
  const damaged = record();
  const raw = damaged.document as ReturnType<typeof document>;
  raw.stacks.control[0]!.adapter.weight = Number.NaN;
  const malformed = await prepareCanvasRecall({
    metadata: { canvas_v3_metadata: damaged },
    model: main,
    models,
    owner: captureAccountScope(),
  });
  expect(malformed.document?.stacks.control).toEqual([]);
  expect(malformed.partial).toBe(true);
  expect(malformed.document?.stacks.raster).toHaveLength(1);
});

const standaloneControl = () => ({
  control_model: { base: 'sdxl', type: 'controlnet', hash: 'control-hash', key: 'old-install', name: 'Edges' },
  image: { image_name: 'original.png' },
  processed_image: { image_name: 'processed.png' },
  control_weight: 0.65,
  begin_step_percent: 0.15,
  end_step_percent: 0.85,
  control_mode: 'more_control',
});

it.each(['all', 'remix'] as const)(
  'recalls standalone name-only controls through Gallery %s with actual map dimensions and a restore point',
  async (kind) => {
    const store = createWorkbenchStore();
    const target = targetFor(store);
    target.commands.canvas.apply(target.projectId, { type: 'replaceCanvasDocument', document: document() });
    const before = target.getProject()!.canvas.document;
    const t2i = { base: 'sdxl', type: 't2i_adapter', key: 't2i', name: 'Sketch' } as ModelConfig;
    api.metadata.mockResolvedValue({
      width: 1024,
      height: 768,
      controlnets: [standaloneControl()],
      t2iAdapters: [
        {
          t2i_adapter_model: t2i,
          image: { image_name: 'sketch.png' },
          weight: 0.4,
          begin_step_percent: 0,
          end_step_percent: 0.6,
        },
      ],
    });
    api.resolveMany.mockImplementation((names: string[]) =>
      Promise.resolve(names.map((imageName) => ({ imageName, width: 512, height: 256 })))
    );
    expect(
      await executeImageRecall({
        commands: store.commands,
        canvasTarget: target,
        getGenerateValues: () => ({ modelKey: main.key }),
        image: { imageName: 'legacy.png', width: 1024, height: 768 } as GalleryImage,
        kind,
        models: [...models, t2i],
        projectId: target.projectId,
        queryClient: new QueryClient(),
        t,
      })
    ).toBe(true);
    expect(api.resolveMany).toHaveBeenCalledExactlyOnceWith(['processed.png', 'sketch.png'], expect.any(AbortSignal));
    const after = target.getProject()!.canvas;
    expect(after.document.stacks.control).toEqual([
      expect.objectContaining({
        source: { type: 'image', image: { imageName: 'processed.png', width: 512, height: 256 } },
        transform: { x: -128, y: 96, rotation: 0, scaleX: 1.5, scaleY: 4 },
        adapter: {
          model: controlModel.key,
          kind: 'controlnet',
          weight: 0.65,
          beginEndStepPct: [0.15, 0.85],
          controlMode: 'more_control',
        },
      }),
      expect.objectContaining({
        adapter: expect.objectContaining({ model: 't2i', kind: 't2i_adapter', weight: 0.4, beginEndStepPct: [0, 0.6] }),
      }),
    ]);
    expect(after.document.stacks.raster).toEqual(before.stacks.raster);
    expect(after.snapshots.at(-1)?.document).toEqual(before);
  }
);

it.each([
  ['flux', 'control_lora', 'control_lora'],
  ['anima', 'controlnet', 'anima_lllite'],
  ['qwen-image', 'controlnet', 'qwen_image_control'],
  ['flux2', 'controlnet', 'flux2_control'],
] as const)('infers standalone %s control kind from installed model identity', async (base, type, kind) => {
  const adapter = {
    base,
    type,
    key: 'installed',
    name: 'Installed',
    hash: 'portable',
    cond_in_channels: 3,
  } as ModelConfig;
  const model = { base, key: 'main', name: 'Main', type: 'main', variant: 'dev' } as MainModelConfig;
  api.resolveMany.mockResolvedValue([{ imageName: 'processed.png', width: 640, height: 480 }]);
  const plan = await prepareCanvasRecall({
    metadata: {
      controlnets: [
        {
          ...standaloneControl(),
          control_model: { base, type, hash: 'portable' },
          begin_step_percent: 0,
          end_step_percent: 1,
        },
      ],
    },
    model,
    models: [model, adapter],
    owner: captureAccountScope(),
  });
  expect(plan.problems).toEqual([]);
  expect(plan.document?.stacks.control[0]).toMatchObject({ adapter: { model: 'installed', kind } });
});

it.each([
  { controlnets: [] },
  { controlnets: 'damaged' },
  { controlnets: [{ ...standaloneControl(), control_weight: [0.2, 0.9] }] },
  { controlnets: [{ ...standaloneControl(), resize_mode: 'crop_resize' }] },
  { controlnets: [{ ...standaloneControl(), control_model: { key: 'deleted' } }] },
  { controlnets: [{ ...standaloneControl(), processed_image: { image_name: 'deleted.png' } }] },
])('preserves authored Canvas when standalone controls cannot be restored: %j', async (metadata) => {
  api.resolveMany.mockImplementation((names: string[]) =>
    Promise.resolve(
      names.filter((name) => name !== 'deleted.png').map((imageName) => ({ imageName, width: 320, height: 200 }))
    )
  );
  const store = createWorkbenchStore();
  const target = targetFor(store);
  target.commands.canvas.apply(target.projectId, { type: 'replaceCanvasDocument', document: document() });
  const before = target.getProject()!.canvas;
  const add = vi.spyOn(store.commands.notifications, 'add');
  api.metadata.mockResolvedValue({ ...metadata, positive_prompt: 'still independent' });
  expect(
    await executeImageRecall({
      commands: store.commands,
      canvasTarget: target,
      getGenerateValues: () => ({ modelKey: main.key }),
      image: { imageName: 'legacy.png', width: 1024, height: 768 } as GalleryImage,
      kind: 'all',
      models,
      projectId: target.projectId,
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(true);
  expect(target.getProject()!.canvas).toBe(before);
  expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({
    positivePrompt: 'still independent',
  });
  expect(add).toHaveBeenCalledWith(
    expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
  );
});

it.each([false, true])(
  'keeps modern control records authoritative over standalone aliases (empty=%s)',
  async (empty) => {
    const control = {
      model: controlModel,
      image: { image_name: 'modern.png', width: 300, height: 200 },
      position: { x: -70, y: 90 },
      weight: 0.3,
    };
    api.resolveMany.mockResolvedValue([{ imageName: 'modern.png', width: 600, height: 400 }]);
    const plan = await prepareCanvasRecall({
      metadata: { control_layers: empty ? [] : [control], controlnets: [standaloneControl()] },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    expect(plan.document?.stacks.control).toHaveLength(empty ? 0 : 1);
    if (empty) {
      expect(api.resolveMany).not.toHaveBeenCalled();
    } else {
      expect(api.resolveMany).toHaveBeenCalledExactlyOnceWith(['modern.png'], expect.any(AbortSignal));
      expect(plan.document?.stacks.control[0]).toMatchObject({
        transform: { x: -70, y: 90, scaleX: 1, scaleY: 1 },
        source: { image: { width: 600, height: 400 } },
      });
    }
  }
);

it('bounds standalone asset reads, deduplicates maps and refuses a stale project before the next batch', async () => {
  let current = true;
  const controls = Array.from({ length: 1002 }, (_, index) => ({
    ...standaloneControl(),
    processed_image: { image_name: `map-${index % 1001}.png` },
  }));
  api.resolveMany.mockImplementation((names: string[]) => {
    current = false;
    return Promise.resolve(names.map((imageName) => ({ imageName, width: 64, height: 64 })));
  });
  const plan = await prepareCanvasRecall({
    metadata: { controlnets: controls },
    model: main,
    models,
    owner: captureAccountScope(),
    isCurrent: () => current,
  });
  expect(api.resolveMany).toHaveBeenCalledOnce();
  expect(api.resolveMany.mock.calls[0]![0]).toHaveLength(1000);
  expect(plan.document).toBeNull();
  expect(plan.problems).toContain('canvas-changed');
});

it('reports legacy vector loss as partial instead of replacing authored work', async () => {
  const layer = {
    id: 'raster',
    name: 'Legacy raster',
    isEnabled: true,
    isLocked: false,
    position: { x: 0, y: 0 },
    opacity: 1,
    objects: [{ id: 'image', type: 'image', image: { image_name: 'raster.png', width: 320, height: 200 } }],
  };
  const plan = await prepareCanvasRecall({
    metadata: {
      canvas_v2_metadata: {
        rasterLayers: [layer],
        controlLayers: [],
        inpaintMasks: [],
        regionalGuidance: [],
        vectorLayers: [{ id: 'vector' }],
      },
    },
    model: main,
    models,
    owner: captureAccountScope(),
  });
  expect(plan.document?.stacks.raster).toHaveLength(1);
  expect(plan.partial).toBe(true);
  expect(plan.problems).toContain('unsupported-legacy-layers');
});

it.each(['partial', 'pending'] as const)(
  'fits standalone maps to the active frame while preserving authored work in a %s merge',
  async (mode) => {
    const store = createWorkbenchStore();
    const target = targetFor(store);
    target.commands.canvas.apply(target.projectId, { type: 'replaceCanvasDocument', document: document() });
    if (mode === 'pending') {
      store.commands.canvas.appendStagingCandidate({
        projectId: target.projectId,
        candidate: {
          imageName: 'staged.png',
          imageUrl: '',
          thumbnailUrl: '',
          width: 100,
          height: 100,
          queuedAt: '',
          sourceQueueItemId: '',
          placement: { x: 0, y: 0, width: 100, height: 100, opacity: 1 },
        },
      });
    }
    const before = target.getProject()!.canvas;
    api.resolveMany.mockResolvedValue([{ imageName: 'processed.png', width: 256, height: 512 }]);
    const controls = [
      standaloneControl(),
      ...(mode === 'partial' ? [{ ...standaloneControl(), control_weight: [0.2, 0.9] }] : []),
    ];
    const plan = await prepareCanvasRecall({
      metadata: { width: 1024, height: 768, controlnets: controls },
      model: main,
      models,
      owner: captureAccountScope(),
    });
    expect(
      commitCanvasRecall({ plan, target, expectedDocument: before.document, restorePointName: 'Before recall' })
    ).toBe(true);
    const after = target.getProject()!.canvas;
    expect(after.document.bbox).toEqual(before.document.bbox);
    expect(after.document.stacks.raster).toEqual(before.document.stacks.raster);
    expect(after.document.stacks.control).toHaveLength(2);
    expect(
      after.document.stacks.control.find(
        (node) =>
          node.type === 'control' && node.source.type === 'image' && node.source.image.imageName === 'processed.png'
      )
    ).toMatchObject({
      transform: { x: -128, y: 96, rotation: 0, scaleX: 3, scaleY: 2 },
    });
    expect(after.stagingArea).toEqual(before.stagingArea);
    expect(after.snapshots).toHaveLength(0);
    expect(plan.partial).toBe(true);
  }
);

it('preserves authored controls for a malformed modern list rather than interpreting it as an empty list', async () => {
  const plan = await prepareCanvasRecall({
    metadata: { control_layers: null, controlnets: [standaloneControl()] },
    model: main,
    models,
    owner: captureAccountScope(),
  });
  expect(plan.document).toBeNull();
  expect(plan.problems).toContain('invalid-control');
  expect(api.resolveMany).not.toHaveBeenCalled();
});

it('keeps Generate recall independent when the second standalone-control asset batch fails', async () => {
  const store = createWorkbenchStore();
  const target = targetFor(store);
  const before = target.getProject()!.canvas;
  api.metadata.mockResolvedValue({
    positive_prompt: 'independent',
    controlnets: Array.from({ length: 1001 }, (_, index) => ({
      ...standaloneControl(),
      processed_image: { image_name: `map-${index}.png` },
    })),
  });
  api.resolveMany
    .mockImplementationOnce((names: string[]) =>
      Promise.resolve(names.map((imageName) => ({ imageName, width: 64, height: 64 })))
    )
    .mockRejectedValueOnce(new ApiError('Unavailable', 503));
  expect(
    await executeImageRecall({
      commands: store.commands,
      canvasTarget: target,
      getGenerateValues: () => ({ modelKey: main.key }),
      image: { imageName: 'legacy.png', width: 512, height: 512 } as GalleryImage,
      kind: 'remix',
      models,
      queryClient: new QueryClient(),
      projectId: target.projectId,
      t,
    })
  ).toBe(true);
  expect(target.getProject()!.canvas).toBe(before);
  expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({ positivePrompt: 'independent' });
  expect(api.resolveMany.mock.calls.map(([names]) => names.length)).toEqual([1000, 1]);
});

it('stops reference hydration after the originating project changes before the second batch or any commit', async () => {
  const store = createWorkbenchStore();
  const target = targetFor(store);
  const ipModel = { base: 'sdxl', type: 'ip_adapter', key: 'ip', name: 'IP' } as ModelConfig;
  api.metadata.mockResolvedValue({
    positive_prompt: 'stale',
    ipAdapters: Array.from({ length: 1001 }, (_, index) => ({
      ip_adapter_model: ipModel,
      image: { image_name: `ref-${index}.png` },
      weight: 0.5,
      method: 'full',
      clip_vision_model: 'ViT-H',
      begin_step_percent: 0,
      end_step_percent: 1,
    })),
  });
  const setSettings = vi.spyOn(store.commands.generation, 'setSettings');
  const before = target.getProject()!.canvas;
  api.resolveMany.mockImplementation((names: string[]) => {
    store.commands.projects.create();
    return Promise.resolve(names.map((imageName) => ({ imageName, width: 64, height: 64 })));
  });
  expect(
    await executeImageRecall({
      commands: store.commands,
      canvasTarget: {
        ...target,
        getProject: () => (store.queries.isActiveProject(target.projectId) ? target.getProject() : null),
      },
      getGenerateValues: () => ({ modelKey: main.key }),
      image: { imageName: 'legacy.png', width: 512, height: 512 } as GalleryImage,
      kind: 'all',
      models: [...models, ipModel],
      queryClient: new QueryClient(),
      projectId: target.projectId,
      t,
    })
  ).toBe(false);
  expect(api.resolveMany).toHaveBeenCalledOnce();
  expect(setSettings).not.toHaveBeenCalled();
  expect(target.getProject()!.canvas).toBe(before);
});

it.each([
  ['array', 'all'],
  ['array', 'remix'],
  ['entities', 'all'],
  ['entities', 'remix'],
] as const)(
  'warns when Gallery %s historical references are refused during %s, preserving existing references and Canvas',
  async (shape, kind) => {
    const store = createWorkbenchStore();
    const target = targetFor(store);
    const before = target.getProject()!.canvas;
    const ipModel = { base: 'sdxl', key: 'ip', name: 'IP', type: 'ip_adapter' } as ModelConfig;
    const currentReference = {
      id: 'current',
      isEnabled: true,
      config: {
        type: 'ip_adapter',
        model: ipModel,
        image: { original: { image: { image_name: 'current.png', width: 100, height: 100 } } },
        weight: 0.5,
        beginEndStepPct: [0, 1],
        clipVisionModel: 'ViT-H',
        method: 'full',
      },
    };
    const refused = { id: 'old', ipAdapter: { type: 'ip_adapter', image: null } };
    api.metadata.mockResolvedValue({
      positive_prompt: 'still recallable',
      canvas_v2_metadata: {
        rasterLayers: [],
        controlLayers: [],
        inpaintMasks: [],
        regionalGuidance: [],
        referenceImages: shape === 'array' ? [refused] : { entities: [refused] },
      },
    });
    const add = vi.spyOn(store.commands.notifications, 'add');
    expect(
      await executeImageRecall({
        commands: store.commands,
        canvasTarget: target,
        getGenerateValues: () => ({ modelKey: main.key, referenceImages: [currentReference] }),
        image: { imageName: 'historical.png', width: 512, height: 512 } as GalleryImage,
        kind,
        models: [...models, ipModel],
        queryClient: new QueryClient(),
        projectId: target.projectId,
        t,
      })
    ).toBe(true);
    expect(target.getProject()!.canvas).toBe(before);
    expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({
      positivePrompt: 'still recallable',
      referenceImages: [currentReference],
    });
    expect(api.resolveMany).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'widgets.gallery.itemActions.recall.partiallyRecalled',
        message: 'widgets.gallery.itemActions.recall.referenceImagesPartial',
      })
    );
  }
);

const convertedPaintRecall = (partial = true) => {
  const recalled = createEmptyCanvasDocument();
  const paint = (id: string) => ({
    ...base,
    id,
    name: id,
    type: 'raster' as const,
    source: { type: 'paint' as const, bitmap: { imageName: `${id}.png`, width: 64, height: 64 } },
  });
  recalled.stacks.raster = [
    paint('created-a'),
    paint('created-b'),
    {
      ...base,
      id: 'borrowed',
      type: 'raster',
      source: { type: 'image', image: { imageName: 'borrowed.png', width: 64, height: 64 } },
    },
  ];
  if (partial) {
    recalled.stacks.control = [
      {
        ...control(),
        id: 'rejected',
        adapter: { ...control().adapter, model: 'incompatible' },
        source: { type: 'paint' as const, bitmap: { imageName: 'created-rejected.png', width: 64, height: 64 } },
      },
    ];
  }
  return {
    record: {
      version: 1,
      document: recalled,
      controlModels: { rejected: { ...controlModel, base: 'sd-1', key: 'incompatible' } },
    },
    problems: [],
    createdImageNames: ['created-a.png', 'created-b.png', ...(partial ? ['created-rejected.png'] : [])],
  };
};

it.each(['project', 'account', 'promotion-failure', 'project-promotion-failure'] as const)(
  'adopts only retained conversion outputs after all recall writes when %s changes during promotion',
  async (change) => {
    const store = createWorkbenchStore();
    const target = targetFor(store);
    const owner = captureAccountScope();
    const localization = createInstance();
    await localization.init({
      initAsync: false,
      lng: 'en',
      resources: {
        en: {
          translation: Object.values(
            import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' })
          )[0] as Record<string, unknown>,
        },
      },
    });
    vi.spyOn(CanvasOperations, 'prepareLegacyCanvasMetadata').mockResolvedValue(convertedPaintRecall());
    api.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt', canvas_v2_metadata: {} });
    let resolve!: () => void;
    let reject!: (reason: Error) => void;
    api.makeCanvasAsset.mockImplementationOnce(
      () =>
        new Promise<void>((yes, no) => {
          resolve = yes;
          reject = no;
        })
    );
    const add = vi.spyOn(store.commands.notifications, 'add');
    const pending = executeImageRecall({
      commands: store.commands,
      canvasTarget: {
        ...target,
        getProject: () => (store.queries.isActiveProject(target.projectId) ? target.getProject() : null),
      },
      getGenerateValues: () => ({ modelKey: main.key, ...getProjectWidgetValues(target.getProject()!, 'generate') }),
      image: { imageName: 'legacy.png', width: 512, height: 512 } as GalleryImage,
      kind: 'all',
      models,
      queryClient: new QueryClient(),
      projectId: target.projectId,
      owner,
      t: localization.t.bind(localization) as TFunction,
    });
    await vi.waitFor(() => expect(api.makeCanvasAsset).toHaveBeenCalledExactlyOnceWith('created-a.png'));
    expect(target.getProject()!.canvas.document.stacks.raster).toHaveLength(3);
    expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({
      positivePrompt: 'recalled prompt',
    });
    // No recalled snapshot may overwrite an edit made during the durability request.
    store.commands.generation.setSettings(
      { ...getDefaultGenerateSettings(main), model: main, steps: 47 },
      target.projectId
    );
    const noticesBefore = add.mock.calls.length;
    if (change === 'project' || change === 'project-promotion-failure') {
      store.commands.projects.create();
    } else if (change === 'account') {
      accountLifecycle.activate('another-account');
    }
    if (change === 'promotion-failure' || change === 'project-promotion-failure') {
      reject(new Error('Promotion unavailable'));
    } else {
      resolve();
    }
    expect(await pending).toBe(true);
    expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({ steps: 47 });
    expect(api.makeCanvasAsset.mock.calls.map(([name]) => name)).toEqual(
      change === 'account' ? ['created-a.png'] : ['created-a.png', 'created-b.png']
    );
    if (change === 'promotion-failure') {
      expect(add).toHaveBeenCalledWith(
        expect.objectContaining({
          message:
            'Canvas was recalled, but some restored paint images are still temporary. Keep the project open or save it before closing to protect them from intermediate cleanup.',
        })
      );
    } else {
      expect(add.mock.calls).toHaveLength(noticesBefore);
    }
  }
);

it.each(['validation', 'commit'] as const)(
  'leaves prepared pixels reclaimable after a failed %s while recalling independent Generate fields',
  async (failure) => {
    const store = createWorkbenchStore();
    const target = targetFor(store);
    const previous = target.getProject()!.canvas.document;
    const converted = convertedPaintRecall(false);
    vi.spyOn(CanvasOperations, 'prepareLegacyCanvasMetadata').mockResolvedValue(
      failure === 'validation'
        ? { ...converted, record: { ...converted.record, document: { ...converted.record.document, version: 999 } } }
        : converted
    );
    if (failure === 'commit') {
      const apply = store.commands.canvas.apply;
      vi.spyOn(store.commands.canvas, 'apply').mockImplementation((projectId, mutation, origin) =>
        mutation.type === 'replaceCanvasDocument' ? false : apply(projectId, mutation, origin)
      );
    }
    api.metadata.mockResolvedValue({ positive_prompt: 'independent prompt', canvas_v2_metadata: {} });
    expect(
      await executeImageRecall({
        commands: store.commands,
        canvasTarget: target,
        getGenerateValues: () => ({ modelKey: main.key }),
        image: { imageName: 'legacy.png', width: 512, height: 512 } as GalleryImage,
        kind: 'all',
        models,
        queryClient: new QueryClient(),
        projectId: target.projectId,
        t,
      })
    ).toBe(true);
    expect(target.getProject()!.canvas.document).toBe(previous);
    expect(getProjectWidgetValues(target.getProject()!, 'generate')).toMatchObject({
      positivePrompt: 'independent prompt',
    });
    expect(api.makeCanvasAsset).not.toHaveBeenCalled();
  }
);
