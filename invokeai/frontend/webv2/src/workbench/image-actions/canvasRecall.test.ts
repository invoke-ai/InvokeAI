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
import { recordCanvasMetadata } from '@workbench/canvas-operations/saveCanvasToGallery';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ metadata: vi.fn(), resolveMany: vi.fn() }));
vi.mock('@features/gallery', async (original) => ({
  ...(await original<typeof GalleryModule>()),
  galleryImages: api,
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
  api.resolveMany.mockImplementation((names: string[]) => Promise.resolve(names.map((imageName) => ({ imageName }))));
});

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
