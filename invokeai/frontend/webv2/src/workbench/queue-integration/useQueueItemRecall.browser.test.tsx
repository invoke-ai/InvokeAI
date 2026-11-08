import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type * as ModelsModule from '@features/models';
import type { QueueGenerationMeta } from '@features/queue/contracts';
import type * as CanvasOperationsModule from '@workbench/canvas-operations/api';
import type { ImageRecallKind } from '@workbench/image-actions';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { galleryDurability, galleryImages } from '@features/gallery';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { createDefaultVideoWidgetValues } from '@features/video';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { recordCanvasMetadata } from '@workbench/canvas-operations/saveCanvasToGallery';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { LocalRecallSnapshot } from './useLocalRecallSnapshot';

import { useQueueItemRecall } from './useQueueItemRecall';

const mocks = vi.hoisted(() => ({
  notifyInfo: vi.fn(),
  notifySuccess: vi.fn(),
  openWidget: vi.fn(),
  patchValues: vi.fn(),
  setSettings: vi.fn(),
  prepareLegacy: vi.fn(),
  models: [] as ModelsModule.ModelConfig[],
  store: null as ReturnType<typeof createWorkbenchStore> | null,
  snapshot: undefined as LocalRecallSnapshot | null | undefined,
}));

vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsModule>()),
  prepareLegacyCanvasMetadata: mocks.prepareLegacy,
}));
vi.mock('./useLocalRecallSnapshot', () => ({ useLocalRecallSnapshot: () => mocks.snapshot }));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchQueries: () =>
    mocks.store?.queries ?? {
      getSnapshot: () => ({ activeProject: { id: 'active-project' }, scopedDefaults: {} }),
      getProject: () => ({ id: 'active-project', widgetInstances: {} }),
      isActiveProject: () => true,
    },
  useWorkbenchCommands: () =>
    mocks.store?.commands ?? {
      generation: { setSettings: mocks.setSettings },
      widgets: { patchValues: mocks.patchValues },
    },
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsModule>()),
  ensureModelsLoaded: () => Promise.resolve(),
  useModelsSelector: () => mocks.models,
}));
vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: mocks.notifyInfo, success: mocks.notifySuccess }),
}));
vi.mock('@workbench/useOpenWorkbenchWidget', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useOpenWorkbenchWidget: () => mocks.openWidget,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

seedArchitectureCapabilities();

const NO_META: QueueGenerationMeta = {};
const generateSnapshot = {
  positivePrompt: 'snapshot prompt',
  seed: 7,
  seedMode: 'fixed',
} as GenerateWidgetValues;

const makeCanvasMetadata = () => {
  const document = createEmptyCanvasDocument();
  document.stacks.raster = [
    {
      id: 'recorded-image',
      name: 'Recorded image',
      type: 'raster',
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      blendMode: 'normal',
      transform: { x: 0, y: 0, rotation: 0, scaleX: 1, scaleY: 1 },
      source: { type: 'image', image: { imageName: 'recorded.png', width: 640, height: 480 } },
    },
  ];
  return { canvas_v3_metadata: recordCanvasMetadata(document, { models: [] }) };
};

type Handle = ReturnType<typeof useQueueItemRecall>;
const handle = createRef<Handle>();
const Probe = ({ meta, ref }: { meta: QueueGenerationMeta; ref: Ref<Handle> }) => {
  const value = useQueueItemRecall('webv2:p:project-1:q:item-1', meta);
  useImperativeHandle(ref, () => value, [value]);
  return null;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (snapshot: LocalRecallSnapshot | null | undefined, meta: QueueGenerationMeta = NO_META) => {
  mocks.snapshot = snapshot;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root?.render(<Probe meta={meta} ref={handle} />));
};
const recall = (kind: ImageRecallKind) => act(() => handle.current!.recall(kind));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.store = null;
  mocks.models = [];
});
afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.restoreAllMocks();
});

describe('useQueueItemRecall', () => {
  it('offers nothing and stays inert while the durable snapshot lookup is pending', async () => {
    await render(undefined);
    expect(Object.values(handle.current!.capabilities).some(Boolean)).toBe(false);
    await recall('all');
    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).not.toHaveBeenCalled();
  });

  it('recalls a generate submission into the Generate panel and reveals it', async () => {
    await render({ generateValues: generateSnapshot, sourceId: 'canvas' });
    expect(handle.current!.capabilities).toMatchObject({ all: true, prompts: true, seed: true });

    await recall('all');

    expect(mocks.setSettings).toHaveBeenCalledWith(generateSnapshot, 'active-project');
    expect(mocks.openWidget).toHaveBeenCalledWith('generate', { preferredRegions: ['left'] });
    expect(mocks.notifySuccess).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.settingsRecalledDescription');
  });

  it('patches a video submission into the Video panel instead', async () => {
    const videoValues = { ...createDefaultVideoWidgetValues(), positivePrompt: 'video prompt' };
    await render({ sourceId: 'video', videoValues });

    await recall('all');

    expect(mocks.patchValues).toHaveBeenCalledWith('video', videoValues);
    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.openWidget).toHaveBeenCalledWith('video', { preferredRegions: ['left'] });
  });

  it('explains an unavailable recall instead of writing anything', async () => {
    // A foreign item: no snapshot, only the executed prompt from the session.
    await render(null, { positivePrompt: 'session prompt' });
    expect(handle.current!.capabilities).toMatchObject({ all: false, prompts: true, seed: false });

    await recall('seed');

    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.recallUnavailable');
  });

  it('names the missing capability table, not a missing model, while the table is absent', async () => {
    await render(null, { positivePrompt: 'session prompt' });

    await act(() => resetArchitectureCapabilities());
    try {
      await recall('seed');
    } finally {
      await act(() => setArchitectureCapabilities(architectureCapabilitiesFixture));
    }

    expect(mocks.setSettings).not.toHaveBeenCalled();
    expect(mocks.notifyInfo).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.recallUnavailableCapabilities');
  });
});

it('recalls a canvas queue snapshot through the real document owner and keeps its authored controls', async () => {
  accountLifecycle.activate('queue-canvas-recall');
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  const main = { base: 'sdxl', key: 'queue-main', name: 'SDXL', type: 'main' } as MainModelConfig &
    ModelsModule.ModelConfig;
  const controlModel = {
    base: 'sdxl',
    key: 'queue-control',
    name: 'Edges',
    type: 'controlnet',
  } as ModelsModule.ModelConfig;
  mocks.models = [main, controlModel];
  mocks.store = createWorkbenchStore();
  const document = createEmptyCanvasDocument();
  document.stacks.control = [
    {
      id: 'submitted-control',
      name: 'Recorded edges',
      type: 'control',
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      blendMode: 'normal',
      transform: { x: -20, y: 40, rotation: 0, scaleX: 1, scaleY: 1 },
      source: { type: 'image', image: { imageName: 'edges.png', width: 640, height: 480 } },
      withTransparencyEffect: false,
      adapter: {
        kind: 'controlnet',
        model: controlModel.key,
        weight: 0.65,
        beginEndStepPct: [0.1, 0.9],
        controlMode: 'balanced',
      },
    },
  ];
  vi.spyOn(galleryImages, 'resolveMany').mockResolvedValue([{ imageName: 'edges.png' }] as Awaited<
    ReturnType<typeof galleryImages.resolveMany>
  >);
  await render({
    sourceId: 'canvas',
    generateValues: { ...getDefaultGenerateSettings(main), model: main, modelKey: main.key },
    canvasMetadata: {
      canvas_v3_metadata: recordCanvasMetadata(document, { models: [controlModel], settings: { strength: 0.4 } }),
    },
  });
  await recall('all');
  expect(mocks.store.queries.getSnapshot().activeProject.canvas.document.stacks.control[0]).toMatchObject({
    name: 'Recorded edges',
    transform: { x: -20, y: 40 },
    adapter: { model: 'queue-control', weight: 0.65 },
  });
  expect(mocks.store.queries.getSnapshot().activeProject.canvas.snapshots).toHaveLength(1);
});

it('retains prompt tool edits made while canvas assets load and recalls the snapshot model', async () => {
  accountLifecycle.activate('queue-canvas-concurrent-recall');
  const main = { base: 'sdxl', key: 'queue-main', name: 'SDXL', type: 'main' } as MainModelConfig &
    ModelsModule.ModelConfig;
  const other = { base: 'sd-1', key: 'other-main', name: 'Other', type: 'main' } as MainModelConfig &
    ModelsModule.ModelConfig;
  mocks.models = [main, other];
  const store = (mocks.store = createWorkbenchStore());
  const projectId = store.queries.getSnapshot().activeProject.id;
  store.commands.generation.setSettings(
    { ...getDefaultGenerateSettings(main), model: main, modelKey: main.key },
    projectId
  );
  let resolveAssets!: (images: Awaited<ReturnType<typeof galleryImages.resolveMany>>) => void;
  const assets = new Promise<Awaited<ReturnType<typeof galleryImages.resolveMany>>>((resolve) => {
    resolveAssets = resolve;
  });
  const resolveMany = vi.spyOn(galleryImages, 'resolveMany').mockReturnValue(assets);
  await render({
    sourceId: 'canvas',
    generateValues: {
      ...getDefaultGenerateSettings(main),
      model: main,
      modelKey: main.key,
      positivePrompt: 'Recalled',
    },
    canvasMetadata: makeCanvasMetadata(),
  });
  let pending!: Promise<void>;
  await act(() => {
    pending = Promise.resolve(handle.current!.recall('remix'));
  });
  await vi.waitFor(() => expect(resolveMany).toHaveBeenCalled());
  store.commands.generation.patchSettings(
    {
      expandPromptModelKey: 'new-prompt-tool',
      imageToPromptModelKey: 'new-vision-tool',
      model: other,
      modelKey: other.key,
    },
    projectId
  );
  await act(async () => {
    resolveAssets([{ imageName: 'recorded.png' }] as Awaited<ReturnType<typeof galleryImages.resolveMany>>);
    await pending;
  });
  const snapshot = store.queries.getSnapshot();
  expect(getResolvedProjectWidgetValues(snapshot.activeProject, 'generate', snapshot.scopedDefaults)).toMatchObject({
    expandPromptModelKey: 'new-prompt-tool',
    imageToPromptModelKey: 'new-vision-tool',
    modelKey: main.key,
    positivePrompt: 'Recalled',
    seedMode: 'random',
  });
  expect(snapshot.activeProject.canvas.document.stacks.raster[0]).toMatchObject({ name: 'Recorded image' });
});

it('recalls Generate settings and reports a partial result when canvas assets cannot load', async () => {
  accountLifecycle.activate('queue-canvas-failed-recall');
  const main = { base: 'sdxl', key: 'queue-main', name: 'SDXL', type: 'main' } as MainModelConfig &
    ModelsModule.ModelConfig;
  mocks.models = [main];
  const store = (mocks.store = createWorkbenchStore());
  const original = store.queries.getSnapshot().activeProject.canvas.document;
  vi.spyOn(galleryImages, 'resolveMany').mockRejectedValue(new Error('Asset lookup unavailable'));
  await render({
    sourceId: 'canvas',
    generateValues: {
      ...getDefaultGenerateSettings(main),
      model: main,
      modelKey: main.key,
      positivePrompt: 'Recalled',
    },
    canvasMetadata: makeCanvasMetadata(),
  });
  await recall('all');
  const snapshot = store.queries.getSnapshot();
  expect(
    getResolvedProjectWidgetValues(snapshot.activeProject, 'generate', snapshot.scopedDefaults).positivePrompt
  ).toBe('Recalled');
  expect(snapshot.activeProject.canvas.document).toBe(original);
  expect(mocks.notifyInfo).toHaveBeenCalledWith(expect.any(String), 'widgets.gallery.itemActions.recall.canvasPartial');
  expect(mocks.notifySuccess).toHaveBeenCalledWith(expect.any(String), 'widgets.queue.settingsRecalledDescription');
});

it.each(['account', 'project', 'canvas'] as const)(
  'fences a canvas queue recall after its %s changes during asset lookup',
  async (change) => {
    accountLifecycle.activate('queue-canvas-fenced-recall');
    const main = { base: 'sdxl', key: 'queue-main', name: 'SDXL', type: 'main' } as MainModelConfig &
      ModelsModule.ModelConfig;
    mocks.models = [main];
    const store = (mocks.store = createWorkbenchStore());
    const projectId = store.queries.getSnapshot().activeProject.id;
    let resolveAssets!: (images: Awaited<ReturnType<typeof galleryImages.resolveMany>>) => void;
    const assets = new Promise<Awaited<ReturnType<typeof galleryImages.resolveMany>>>((resolve) => {
      resolveAssets = resolve;
    });
    const resolveMany = vi.spyOn(galleryImages, 'resolveMany').mockReturnValue(assets);
    await render({
      sourceId: 'canvas',
      generateValues: {
        ...getDefaultGenerateSettings(main),
        model: main,
        modelKey: main.key,
        positivePrompt: 'Recalled',
      },
      canvasMetadata: makeCanvasMetadata(),
    });
    let pending!: Promise<void>;
    await act(() => {
      pending = Promise.resolve(handle.current!.recall('all'));
    });
    await vi.waitFor(() => expect(resolveMany).toHaveBeenCalled());
    if (change === 'account') {
      accountLifecycle.activate('next-account');
    } else if (change === 'project') {
      store.commands.projects.create();
    } else {
      store.commands.canvas.apply(projectId, {
        type: 'setCanvasBbox',
        bbox: { x: 100, y: 200, width: 512, height: 512 },
      });
    }
    const original = store.queries.getProject(projectId)!;
    await act(async () => {
      resolveAssets([{ imageName: 'recorded.png' }] as Awaited<ReturnType<typeof galleryImages.resolveMany>>);
      await pending;
    });
    const project = store.queries.getProject(projectId)!;
    expect(project.canvas.document).toBe(original.canvas.document);
    expect(project.canvas.snapshots).toHaveLength(0);
    if (change === 'canvas') {
      expect(
        getResolvedProjectWidgetValues(project, 'generate', store.queries.getSnapshot().scopedDefaults).positivePrompt
      ).toBe('Recalled');
      expect(mocks.notifyInfo).toHaveBeenCalledWith(
        expect.any(String),
        'widgets.gallery.itemActions.recall.canvasPartial'
      );
    } else {
      expect(project).toBe(original);
      expect(mocks.notifyInfo).not.toHaveBeenCalled();
      expect(mocks.notifySuccess).not.toHaveBeenCalled();
      expect(mocks.openWidget).not.toHaveBeenCalled();
    }
  }
);

it.each(['project', 'account', 'promotion-failure', 'project-promotion-failure'] as const)(
  'finishes Queue recall writes and navigation before adoption when %s changes during promotion',
  async (change) => {
    accountLifecycle.activate('queue-paint-adoption');
    const main = { base: 'sdxl', key: 'main', name: 'SDXL', type: 'main' } as MainModelConfig;
    mocks.models = [main as ModelsModule.ModelConfig];
    mocks.store = createWorkbenchStore();
    const projectId = mocks.store.queries.getSnapshot().activeProject.id;
    const document = createEmptyCanvasDocument();
    document.stacks.raster = ['created-a.png', 'created-b.png'].map((imageName, index) => ({
      id: `paint-${index}`,
      name: 'Paint',
      type: 'raster',
      isEnabled: true,
      isLocked: false,
      opacity: 1,
      blendMode: 'normal',
      transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 },
      source: { type: 'paint', bitmap: { imageName, width: 64, height: 64 } },
    }));
    mocks.prepareLegacy.mockResolvedValue({
      record: { version: 1, document },
      problems: [],
      createdImageNames: ['created-a.png', 'created-b.png'],
    });
    vi.spyOn(galleryImages, 'resolveMany').mockResolvedValue(
      ['created-a.png', 'created-b.png'].map((imageName) => ({ imageName })) as Awaited<
        ReturnType<typeof galleryImages.resolveMany>
      >
    );
    let resolve!: () => void;
    let reject!: (error: Error) => void;
    const promote = vi
      .spyOn(galleryDurability, 'makeCanvasAsset')
      .mockResolvedValue(undefined)
      .mockImplementationOnce(
        () =>
          new Promise<void>((yes, no) => {
            resolve = yes;
            reject = no;
          })
      );
    await render({
      sourceId: 'canvas',
      generateValues: {
        ...getDefaultGenerateSettings(main),
        positivePrompt: 'Queue recall',
        model: main,
        modelKey: main.key,
      },
      canvasMetadata: { canvas_v2_metadata: {} },
    });
    await act(async () => {
      const pending = handle.current!.recall('all');
      await vi.waitFor(() => expect(promote).toHaveBeenCalledExactlyOnceWith('created-a.png'));
      expect(getResolvedProjectWidgetValues(mocks.store!.queries.getProject(projectId)!, 'generate', {})).toMatchObject(
        { positivePrompt: 'Queue recall' }
      );
      expect(mocks.openWidget).toHaveBeenCalledOnce();
      expect(mocks.notifySuccess).toHaveBeenCalledOnce();
      mocks.store!.commands.generation.setSettings(
        { ...getDefaultGenerateSettings(main), model: main, steps: 47 },
        projectId
      );
      const infosBefore = mocks.notifyInfo.mock.calls.length;
      if (change === 'project' || change === 'project-promotion-failure') {
        mocks.store!.commands.projects.create();
      } else if (change === 'account') {
        accountLifecycle.activate('other-account');
      }
      if (change === 'promotion-failure' || change === 'project-promotion-failure') {
        reject(new Error('Promotion unavailable'));
      } else {
        resolve();
      }
      await pending;
      await vi.waitFor(() => expect(promote).toHaveBeenCalledTimes(change === 'account' ? 1 : 2));
      expect(mocks.openWidget).toHaveBeenCalledOnce();
      expect(mocks.notifySuccess).toHaveBeenCalledOnce();
      expect(getResolvedProjectWidgetValues(mocks.store!.queries.getProject(projectId)!, 'generate', {})).toMatchObject(
        { steps: 47 }
      );
      if (change === 'promotion-failure') {
        await vi.waitFor(() =>
          expect(mocks.notifyInfo).toHaveBeenCalledWith(
            expect.any(String),
            'widgets.gallery.itemActions.recall.canvasImagesTemporary'
          )
        );
      } else {
        expect(mocks.notifyInfo.mock.calls).toHaveLength(infosBefore);
      }
    });
  }
);
