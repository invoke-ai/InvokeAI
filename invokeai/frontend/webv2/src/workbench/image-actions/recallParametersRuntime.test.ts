import type { ComponentModelConfig } from '@features/generation/contracts';
import type { ModelsSnapshot } from '@features/models';
import type { SocketHub } from '@platform/transport/socketHub';
import type { TFunction } from 'i18next';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: {
    metadata: vi.fn(),
    resolveMany: vi.fn(),
  },
}));
const modelsApi = vi.hoisted(() => ({
  ensureModelsLoaded: vi.fn(),
  getModelsSnapshot: vi.fn(),
}));

vi.mock('@features/gallery', () => galleryApi);
vi.mock('@features/models', () => modelsApi);

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';

import { createRecallParametersRuntime } from './recallParametersRuntime';

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
    emit: (event: string, payload: unknown) => {
      for (const handler of handlers.get(event) ?? []) {
        handler(payload as never);
      }
    },
    handlerCount: () => [...handlers.values()].reduce((count, eventHandlers) => count + eventHandlers.size, 0),
    hub,
  };
};

const sdxl = { base: 'sdxl', hash: 'h', key: 'sdxl-main', name: 'SDXL', type: 'main' } as ComponentModelConfig;
const image = (imageName: string) => ({ height: 512, image_name: imageName, width: 512 });
const event = (parameters: Record<string, unknown>, userId = 'owner') => ({
  parameters,
  queue_id: 'default',
  user_id: userId,
});

const flush = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

const snapshot = (overrides: Partial<ModelsSnapshot> = {}): ModelsSnapshot =>
  ({ error: null, models: [sdxl], status: 'loaded', ...overrides }) as ModelsSnapshot;

const referenceNames = (values: Record<string, unknown>) =>
  (values.referenceImages as { config: { image: { original: { image: { image_name: string } } } } }[]).map(
    (reference) => reference.config.image.original.image.image_name
  );

const promptOf = (store: ReturnType<typeof createWorkbenchStore>, projectId: string) => {
  const project = store.queries.getProject(projectId);
  return project ? getProjectWidgetValues(project, 'generate').positivePrompt : 'project missing';
};

const createRuntime = (
  store: ReturnType<typeof createWorkbenchStore>,
  hub: Pick<SocketHub, 'on'>,
  options: Partial<Parameters<typeof createRecallParametersRuntime>[0]> = {}
  // Nothing registers in this suite, so a reveal can only select or raise an instance already in the layout.
) =>
  createRecallParametersRuntime({
    commands: store.commands,
    hub,
    queries: store.queries,
    reveal: { getWidgetsForRegion: () => [], isEditingText: () => false },
    t,
    ...options,
  });

const t = ((key: string) => key) as unknown as TFunction;

seedArchitectureCapabilities();

describe('createRecallParametersRuntime', () => {
  beforeEach(() => {
    accountLifecycle.activate('owner');
    galleryApi.galleryImages.resolveMany.mockReset();
    galleryApi.galleryImages.resolveMany.mockImplementation((imageNames: string[]) =>
      Promise.resolve(imageNames.map((imageName) => ({ height: 512, imageName, width: 512 })))
    );
    modelsApi.ensureModelsLoaded.mockReset();
    modelsApi.ensureModelsLoaded.mockResolvedValue(undefined);
    modelsApi.getModelsSnapshot.mockReset();
    modelsApi.getModelsSnapshot.mockReturnValue(snapshot());
  });

  it('applies the event to the project that was active when it arrived', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const firstProjectId = store.queries.getSnapshot().activeProject.id;
    let releaseModels: () => void = () => {};
    modelsApi.ensureModelsLoaded.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          releaseModels = resolve;
        })
    );
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'for the first project' }));
    const secondProject = store.commands.projects.create();
    expect(store.queries.getSnapshot().activeProject.id).toBe(secondProject.id);

    await flush();
    releaseModels();
    await flush();

    expect(promptOf(store, firstProjectId)).toBe('for the first project');
    expect(promptOf(store, secondProject.id)).toBeUndefined();
    expect(store.queries.getSnapshot().notifications.at(-1)).toEqual(
      expect.objectContaining({ kind: 'success', title: 'Recalled parameters' })
    );

    runtime.dispose();
  });

  it('applies replayed events first, each to the project recorded for it', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const firstProjectId = store.queries.getSnapshot().activeProject.id;
    const secondProject = store.commands.projects.create();
    const runtime = createRuntime(store, socket.hub, {
      replay: [{ payload: event({ positive_prompt: 'buffered' }), projectId: firstProjectId }],
    });

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'live' }));
    await flush();

    expect(promptOf(store, firstProjectId)).toBe('buffered');
    expect(promptOf(store, secondProject.id)).toBe('live');

    runtime.dispose();
  });

  it('applies events in arrival order so an append sees the replace before it', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ reference_images: [{ image: image('replace.png') }] }));
    socket.emit(
      'recall_parameters_updated',
      event({ append: true, reference_images: [{ image: image('append.png') }] })
    );
    await flush();

    const project = store.queries.getProject(projectId);
    expect(project && referenceNames(getProjectWidgetValues(project, 'generate'))).toEqual([
      'replace.png',
      'append.png',
    ]);

    runtime.dispose();
  });

  it('ignores other users’ events in multi-user mode', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    const runtime = createRuntime(store, socket.hub, { getSessionUserId: () => 'owner' });

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'someone else' }, 'other-user'));
    socket.emit('recall_parameters_updated', event({ positive_prompt: 'mine' }, 'owner'));
    await flush();

    expect(promptOf(store, projectId)).toBe('mine');
    expect(store.queries.getSnapshot().notifications).toHaveLength(1);

    runtime.dispose();
  });

  it('ignores malformed payloads', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', { queue_id: 'default', user_id: 'owner' });
    await flush();

    expect(promptOf(store, projectId)).toBeUndefined();
    expect(store.queries.getSnapshot().notifications).toEqual([]);

    runtime.dispose();
  });

  it('drops work for an account scope that expired before it ran', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'stale' }));
    accountLifecycle.activate('other-account');
    await flush();

    expect(promptOf(store, projectId)).toBeUndefined();
    expect(store.queries.getSnapshot().notifications).toEqual([]);

    runtime.dispose();
  });

  it('detaches on dispose and ignores later events', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    const runtime = createRuntime(store, socket.hub);

    expect(socket.handlerCount()).toBe(1);
    runtime.dispose();
    runtime.dispose();
    expect(socket.handlerCount()).toBe(0);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'after dispose' }));
    await flush();

    expect(promptOf(store, projectId)).toBeUndefined();
  });

  it('reports a model catalog failure and still applies the next event', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    modelsApi.getModelsSnapshot.mockReturnValueOnce(
      snapshot({ error: 'catalog offline', models: [], status: 'error' })
    );
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'unreachable' }));
    await flush();

    expect(promptOf(store, projectId)).toBeUndefined();
    expect(store.queries.getSnapshot().notifications.at(-1)).toEqual(
      expect.objectContaining({ kind: 'error', message: 'catalog offline' })
    );

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'after recovery' }));
    await flush();

    expect(promptOf(store, projectId)).toBe('after recovery');

    runtime.dispose();
  });

  it('keeps the chain alive when applying one event throws', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    modelsApi.ensureModelsLoaded.mockRejectedValueOnce(new Error('unexpected'));
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'first' }));
    socket.emit('recall_parameters_updated', event({ positive_prompt: 'second' }));
    await flush();

    expect(store.queries.getSnapshot().notifications).toContainEqual(
      expect.objectContaining({ kind: 'error', message: 'unexpected' })
    );
    expect(promptOf(store, projectId)).toBe('second');

    runtime.dispose();
  });

  it('brings the Generate widget to the front once a recall applies', async () => {
    const store = createWorkbenchStore();
    const socket = createFakeSocketHub();
    const projectId = store.queries.getSnapshot().activeProject.id;
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'generate' });
    store.commands.widgets.open({ projectId, region: 'left', widgetId: 'video' });
    const activeType = () => {
      const project = store.queries.getProject(projectId)!;
      const active = project.widgetRegions.left.activeInstanceId;
      return active ? project.widgetInstances[active]?.typeId : undefined;
    };
    expect(activeType()).toBe('video');
    const runtime = createRuntime(store, socket.hub);

    socket.emit('recall_parameters_updated', event({ positive_prompt: 'a lighthouse' }));
    await flush();

    expect(activeType()).toBe('generate');

    runtime.dispose();
  });
});
