import type { ModelConfig, ModelsSnapshot } from '@features/models';
import type { SocketHub } from '@platform/transport/socketHub';
import type { RegisteredWidget } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: { metadata: vi.fn(), resolveMany: vi.fn() },
  galleryItems: { resolve: vi.fn() },
  galleryVideos: { metadata: vi.fn() },
}));
const modelsApi = vi.hoisted(() => ({ ensureModelsLoaded: vi.fn(), getModelsSnapshot: vi.fn() }));

vi.mock('@features/gallery', () => galleryApi);
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...modelsApi,
}));

import { createDefaultVideoWidgetValues } from '@features/video';

import { createVideoRecallRuntime } from './videoRecallRuntime';

type SocketHandler = (payload: never) => void;

const createFakeSocketHub = () => {
  const handlers = new Map<string, Set<SocketHandler>>();
  const hub: Pick<SocketHub, 'on'> = {
    on: (event, handler) => {
      const eventHandlers = handlers.get(event) ?? new Set<SocketHandler>();
      eventHandlers.add(handler);
      handlers.set(event, eventHandlers);
      return () => eventHandlers.delete(handler);
    },
  };

  return {
    emit: (payload: unknown) => {
      for (const handler of handlers.get('video_recall_requested') ?? []) {
        handler(payload as never);
      }
    },
    hub,
  };
};

const model = (fields: Record<string, unknown>) =>
  ({
    file_size: 1,
    hash: 'h',
    path: '/m',
    source: 'local',
    source_type: 'path',
    type: 'main',
    ...fields,
  }) as ModelConfig;

const WAN_I2V = model({ base: 'wan', format: 'diffusers', key: 'wan-i2v', name: 'Wan I2V', variant: 'i2v_a14b' });
const WAN_T2V = model({ base: 'wan', format: 'diffusers', key: 'wan-t2v', name: 'Wan T2V', variant: 't2v_a14b' });
const REF2VA = model({
  base: 'minimax-h3',
  format: 'checkpoint',
  key: 'h3-ref2va',
  name: 'H3 Ref2VA',
  variant: 'ref2va',
});

const t = ((key: string) => key) as unknown as TFunction;

const flush = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

const parametersEvent = (parameters: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
  action: 'parameters',
  mode: 'recall',
  parameters,
  queue_id: 'default',
  strict: false,
  user_id: 'owner',
  ...extra,
});

const placementEvent = (action: 'initial_video' | 'reference_video', video: Record<string, unknown> = {}) => ({
  action,
  queue_id: 'default',
  user_id: 'owner',
  video: { duration: 5, fps: 16, height: 480, media_origin: null, video_name: 'clip.mp4', width: 832, ...video },
});

const heldFrame = { height: 480, image_name: 'held.png', width: 832 };

/** The registry's view of the Video widget: placeable on the left, nothing to preload. */
const videoWidget = {
  implementation: { preload: () => undefined },
  manifest: { allowedRegions: ['left'], id: 'video' },
  status: 'enabled',
} as unknown as RegisteredWidget;

const setup = (panelModel: ModelConfig = WAN_I2V) => {
  const store = createWorkbenchStore();
  const socket = createFakeSocketHub();
  const projectId = store.queries.getSnapshot().activeProject.id;

  store.commands.widgets.patchValues(
    'video',
    {
      ...createDefaultVideoWidgetValues([panelModel]),
      ...(panelModel === WAN_I2V ? { firstFrameImage: heldFrame } : {}),
      positivePrompt: 'before',
    },
    projectId
  );
  const runtime = createVideoRecallRuntime({
    commands: store.commands,
    hub: socket.hub,
    queries: store.queries,
    reveal: {
      getWidgetsForRegion: (region) => (region === 'left' ? [videoWidget] : []),
      isEditingText: () => false,
    },
    t,
  });
  /** Whether the Video widget is the visible tab of the project's left region. */
  const videoShown = (id = projectId) => {
    const project = store.queries.getProject(id)!;
    const left = project.widgetRegions.left;

    return (
      !left.isCollapsed &&
      left.activeInstanceId !== null &&
      project.widgetInstances[left.activeInstanceId]?.typeId === 'video'
    );
  };
  const videoValues = (id = projectId) => {
    const project = store.queries.getProject(id);
    return project ? getProjectWidgetValues(project, 'video') : {};
  };
  // Newest first.
  const lastNotice = () => store.queries.getSnapshot().notifications[0];

  return { lastNotice, projectId, runtime, socket, store, videoShown, videoValues };
};

describe('createVideoRecallRuntime', () => {
  beforeEach(() => {
    accountLifecycle.activate('owner');
    modelsApi.ensureModelsLoaded.mockReset();
    modelsApi.ensureModelsLoaded.mockResolvedValue(undefined);
    modelsApi.getModelsSnapshot.mockReset();
    modelsApi.getModelsSnapshot.mockReturnValue({
      error: null,
      models: [WAN_I2V, WAN_T2V, REF2VA],
      status: 'loaded',
    } as ModelsSnapshot);
    galleryApi.galleryItems.resolve.mockReset();
    galleryApi.galleryItems.resolve.mockImplementation(({ name }: { name: string }) =>
      Promise.resolve({ durationSeconds: 5, fps: 16, height: 480, kind: 'video', name, width: 832 })
    );
    galleryApi.galleryImages.resolveMany.mockReset();
    galleryApi.galleryImages.resolveMany.mockImplementation((names: string[]) =>
      Promise.resolve(names.map((imageName) => ({ height: 480, imageName, width: 832 })))
    );
  });

  it('applies a non-strict recall on top of the panel and reveals the Video widget', async () => {
    const { lastNotice, videoShown, runtime, socket, videoValues } = setup();

    socket.emit(parametersEvent({ positive_prompt: 'a heron', seed: 99 }));
    await flush();

    expect(videoValues()).toMatchObject({ firstFrameImage: heldFrame, positivePrompt: 'a heron', seed: 99 });
    expect(lastNotice()).toEqual(expect.objectContaining({ kind: 'success', title: 'Recalled video data' }));
    expect(videoShown()).toBe(true);

    runtime.dispose();
  });

  it('clears media a strict recall does not name', async () => {
    const { runtime, socket, videoValues } = setup();

    socket.emit(parametersEvent({ positive_prompt: 'a heron' }, { strict: true }));
    await flush();

    expect(videoValues()).toMatchObject({ firstFrameImage: null, positivePrompt: 'a heron' });

    runtime.dispose();
  });

  it('titles a remix as one', async () => {
    const { lastNotice, runtime, socket } = setup();

    socket.emit(parametersEvent({ positive_prompt: 'a heron' }, { mode: 'remix' }));
    await flush();

    expect(lastNotice()).toEqual(expect.objectContaining({ title: 'Remixed video' }));

    runtime.dispose();
  });

  it('applies to the project it arrived for, without pulling a different project to the front', async () => {
    const { videoShown, projectId, runtime, socket, store, videoValues } = setup();
    let releaseModels: () => void = () => {};
    modelsApi.ensureModelsLoaded.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          releaseModels = resolve;
        })
    );

    socket.emit(parametersEvent({ positive_prompt: 'for the first project' }));
    const second = store.commands.projects.create();
    await flush();
    releaseModels();
    await flush();

    expect(videoValues(projectId).positivePrompt).toBe('for the first project');
    expect(videoValues(second.id).positivePrompt).toBeUndefined();
    expect(videoShown()).toBe(false);

    runtime.dispose();
  });

  it('sets the initial video as the Initial Video field would', async () => {
    const { lastNotice, videoShown, runtime, socket, videoValues } = setup();

    socket.emit(placementEvent('initial_video'));
    await flush();

    expect(videoValues()).toMatchObject({ firstFrameImage: null, sourceVideo: { fps: 16, video_name: 'clip.mp4' } });
    expect(lastNotice()).toEqual(
      expect.objectContaining({ kind: 'success', title: 'widgets.video.placement.initialVideoSet' })
    );
    expect(videoShown()).toBe(true);

    runtime.dispose();
  });

  it('declines a reference video the panel model cannot take, leaving the panel alone', async () => {
    const { lastNotice, videoShown, runtime, socket, videoValues } = setup();
    const before = videoValues();

    socket.emit(placementEvent('reference_video'));
    await flush();

    expect(videoValues()).toBe(before);
    expect(lastNotice()).toEqual(
      expect.objectContaining({
        kind: 'info',
        message: 'widgets.video.placement.referenceUnsupported',
        title: 'widgets.video.placement.referenceNotAdded',
      })
    );
    expect(videoShown()).toBe(false);

    runtime.dispose();
  });

  it.each([
    ['a parameters event without its strict flag', { ...parametersEvent({}), strict: undefined }],
    ['a placement without the video it places', { ...placementEvent('initial_video'), video: null }],
    ['a placement whose video has no size', { ...placementEvent('initial_video'), video: { video_name: 'clip.mp4' } }],
    ['an unknown action', { ...placementEvent('initial_video'), action: 'delete_video' }],
  ])('ignores %s', async (_label, payload) => {
    const { videoShown, runtime, socket, store, videoValues } = setup();
    const before = videoValues();
    const notices = store.queries.getSnapshot().notifications.length;

    socket.emit(payload);
    await flush();

    expect(videoValues()).toBe(before);
    expect(store.queries.getSnapshot().notifications).toHaveLength(notices);
    expect(videoShown()).toBe(false);

    runtime.dispose();
  });

  describe('partial media', () => {
    const referenceNames = (values: Record<string, unknown>) =>
      (
        values.references as {
          clip?: { video_name: string };
          fromSourceVideo?: boolean;
          image?: { image_name: string };
        }[]
      ).map((entry) =>
        entry.image ? entry.image.image_name : `${entry.clip?.video_name}${entry.fromSourceVideo ? '*' : ''}`
      );

    it('keeps the Ref2VA continuity reference when the references are replaced, and moves it with the clip', async () => {
      const { runtime, socket, videoValues } = setup(REF2VA);

      socket.emit(placementEvent('initial_video'));
      await flush();
      expect(referenceNames(videoValues())).toEqual(['clip.mp4*']);

      socket.emit(parametersEvent({ minimax_h3_references: [{ image_name: 'B.png', kind: 'image' }] }));
      await flush();
      expect(referenceNames(videoValues())).toEqual(['B.png', 'clip.mp4*']);
      expect(videoValues()).toMatchObject({ sourceVideo: { video_name: 'clip.mp4' } });

      socket.emit(parametersEvent({ source_video: { video_name: 'other.mp4' } }));
      await flush();
      expect(referenceNames(videoValues())).toEqual(['B.png', 'other.mp4*']);
      expect(videoValues()).toMatchObject({ sourceVideo: { video_name: 'other.mp4' } });

      runtime.dispose();
    });

    it('removes the initial video when a sent reference list leaves no slot for its continuity reference', async () => {
      const { runtime, socket, videoValues } = setup(REF2VA);

      socket.emit(placementEvent('initial_video'));
      await flush();
      socket.emit(
        parametersEvent({
          minimax_h3_references: ['a.mp4', 'b.mp4', 'c.mp4'].map((video_name) => ({ kind: 'video', video_name })),
        })
      );
      await flush();

      expect(referenceNames(videoValues())).toEqual(['a.mp4', 'b.mp4', 'c.mp4']);
      expect(videoValues().sourceVideo).toBeNull();

      runtime.dispose();
    });

    it('clears the references for an explicitly empty list sent beside an initial video', async () => {
      const { runtime, socket, videoValues } = setup(REF2VA);

      for (const name of ['a.mp4', 'b.mp4', 'c.mp4']) {
        socket.emit(placementEvent('reference_video', { video_name: name }));
        await flush();
      }
      socket.emit(parametersEvent({ minimax_h3_references: [], source_video: { video_name: 'clip.mp4' } }));
      await flush();

      expect(referenceNames(videoValues())).toEqual(['clip.mp4*']);
      expect(videoValues()).toMatchObject({ sourceVideo: { video_name: 'clip.mp4' } });

      runtime.dispose();
    });

    it('ignores a first frame a Ref2VA panel cannot use, keeping its initial video', async () => {
      const { runtime, socket, videoValues } = setup(REF2VA);

      socket.emit(placementEvent('initial_video'));
      await flush();
      socket.emit(parametersEvent({ first_frame_image: { image_name: 'first.png' } }));
      await flush();

      expect(videoValues()).toMatchObject({ firstFrameImage: null, sourceVideo: { video_name: 'clip.mp4' } });
      expect(referenceNames(videoValues())).toEqual(['clip.mp4*']);

      runtime.dispose();
    });

    it('does not let media the panel model cannot use displace what it holds', async () => {
      const { lastNotice, runtime, socket, videoValues } = setup();

      socket.emit(parametersEvent({ minimax_h3_references: [{ image_name: 'B.png', kind: 'image' }] }));
      await flush();

      expect(videoValues()).toMatchObject({ firstFrameImage: heldFrame, references: [] });
      expect(lastNotice()).toEqual(
        expect.objectContaining({ kind: 'info', title: 'widgets.video.externalRecall.nothingApplied' })
      );

      runtime.dispose();
    });
  });

  it('brings a Video widget in a background tab to the front', async () => {
    const { projectId, runtime, socket, store, videoShown } = setup();
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'generate' });
    expect(videoShown()).toBe(false);

    socket.emit(parametersEvent({ positive_prompt: 'a heron' }));
    await flush();

    expect(videoShown()).toBe(true);

    runtime.dispose();
  });

  describe('placements', () => {
    it('appends a reference video with the defaults its wire facts imply', async () => {
      const { lastNotice, runtime, socket, videoValues } = setup(REF2VA);

      socket.emit(placementEvent('reference_video', { duration: 10, fps: 24, media_origin: 'audio_upload' }));
      await flush();

      expect(videoValues().references).toEqual([
        expect.objectContaining({
          clip: expect.objectContaining({ endFrame: 239, fps: 24, numFrames: 240, video_name: 'clip.mp4' }),
          conditioning: 'audio',
          kind: 'video',
        }),
      ]);
      expect(lastNotice()).toEqual(
        expect.objectContaining({ kind: 'success', title: 'widgets.video.placement.referenceAdded' })
      );

      runtime.dispose();
    });

    it('declines an initial video a Ref2VA panel has no reference slot to anchor', async () => {
      const { lastNotice, runtime, socket, videoValues } = setup(REF2VA);

      for (const name of ['a.mp4', 'b.mp4', 'c.mp4']) {
        socket.emit(placementEvent('reference_video', { video_name: name }));
        await flush();
      }
      expect(videoValues().references).toHaveLength(3);
      const before = videoValues();

      socket.emit(placementEvent('initial_video'));
      await flush();

      expect(videoValues()).toBe(before);
      expect(lastNotice()).toEqual(
        expect.objectContaining({ kind: 'info', message: 'widgets.video.placement.initialVideoFull' })
      );

      runtime.dispose();
    });

    it('sets an initial video the panel model cannot extend, and says so', async () => {
      const { lastNotice, runtime, socket, videoValues } = setup(WAN_T2V);

      socket.emit(placementEvent('initial_video'));
      await flush();

      expect(videoValues()).toMatchObject({ sourceVideo: { video_name: 'clip.mp4' } });
      expect(lastNotice()).toEqual(
        expect.objectContaining({ kind: 'info', message: 'widgets.video.placement.initialVideoUnused' })
      );

      runtime.dispose();
    });

    it('places into the project it arrived for', async () => {
      const { projectId, runtime, socket, store, videoValues } = setup();
      let releaseModels: () => void = () => {};
      modelsApi.ensureModelsLoaded.mockImplementation(
        () =>
          new Promise<void>((resolve) => {
            releaseModels = resolve;
          })
      );

      socket.emit(placementEvent('initial_video'));
      const second = store.commands.projects.create();
      await flush();
      releaseModels();
      await flush();

      expect(videoValues(projectId)).toMatchObject({ sourceVideo: { video_name: 'clip.mp4' } });
      expect(videoValues(second.id).sourceVideo).toBeUndefined();

      runtime.dispose();
    });
  });
});
