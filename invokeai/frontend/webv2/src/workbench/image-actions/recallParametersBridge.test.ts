import type { SocketHub } from '@platform/transport/socketHub';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { describe, expect, it, vi } from 'vitest';

import { attachRecallParametersRuntime, attachVideoRecallRuntime } from './recallParametersBridge';

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

type RuntimeOptions = { hub: Pick<SocketHub, 'on'>; replay?: readonly { payload: unknown; projectId: string }[] };

const createDeps = () => {
  const reportError = vi.fn();
  const received: { payload: unknown; projectId: string }[] = [];
  const dispose = vi.fn();
  const active = { id: 'project-1' };
  const createRecallParametersRuntime = vi.fn(({ hub, replay = [] }: RuntimeOptions) => {
    received.push(...replay);
    const detach = hub.on('recall_parameters_updated', (payload: unknown) => {
      received.push({ payload, projectId: active.id });
    });
    dispose.mockImplementation(detach);
    return { dispose };
  });

  return {
    active,
    commands: {
      generation: {} as WorkbenchCommands['generation'],
      notifications: { reportError } as unknown as WorkbenchCommands['notifications'],
      widgets: {} as WorkbenchCommands['widgets'],
    },
    createRecallParametersRuntime,
    dispose,
    queries: {
      getProject: () => null,
      getSnapshot: () => ({ activeProject: active }),
    } as unknown as Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>,
    received,
    reveal: { getWidgetsForRegion: () => [], isEditingText: () => false },
    t: ((key: string) => key) as unknown as TFunction,
    reportError,
  };
};

const event = (parameters: Record<string, unknown>) => ({ parameters, queue_id: 'default', user_id: 'owner' });
const stepsOf = (entry: { payload: unknown }) => (entry.payload as { parameters: { steps: number } }).parameters.steps;

const flush = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

describe('attachRecallParametersRuntime', () => {
  it('loads the runtime on the first event and hands over buffered events with their arrival project', async () => {
    const socket = createFakeSocketHub();
    const deps = createDeps();
    let release: (module: {
      createRecallParametersRuntime: typeof deps.createRecallParametersRuntime;
    }) => void = () => {};
    const load = vi.fn(
      () =>
        new Promise<{ createRecallParametersRuntime: typeof deps.createRecallParametersRuntime }>((resolve) => {
          release = resolve;
        })
    );
    const getSessionUserId = () => 'owner';
    const bridge = attachRecallParametersRuntime({ ...deps, getSessionUserId, hub: socket.hub, load });

    expect(load).not.toHaveBeenCalled();

    socket.emit('recall_parameters_updated', event({ steps: 1 }));
    deps.active.id = 'project-2';
    socket.emit('recall_parameters_updated', event({ steps: 2 }));
    expect(load).toHaveBeenCalledTimes(1);

    release({ createRecallParametersRuntime: deps.createRecallParametersRuntime });
    await flush();
    // The multi-user fence must survive the lazy hand-over: admins receive other users' recall events.
    expect(deps.createRecallParametersRuntime).toHaveBeenCalledWith(expect.objectContaining({ getSessionUserId }));
    socket.emit('recall_parameters_updated', event({ steps: 3 }));

    expect(deps.received.map((entry) => [stepsOf(entry), entry.projectId])).toEqual([
      [1, 'project-1'],
      [2, 'project-2'],
      [3, 'project-2'],
    ]);
    expect(socket.handlerCount()).toBe(1);

    bridge.dispose();
    expect(deps.dispose).toHaveBeenCalledTimes(1);
    expect(socket.handlerCount()).toBe(0);
  });

  it('never creates the runtime when disposed while the module is loading', async () => {
    const socket = createFakeSocketHub();
    const deps = createDeps();
    const bridge = attachRecallParametersRuntime({
      ...deps,
      hub: socket.hub,
      load: () => Promise.resolve({ createRecallParametersRuntime: deps.createRecallParametersRuntime }),
    });

    socket.emit('recall_parameters_updated', event({ steps: 1 }));
    bridge.dispose();
    await flush();

    expect(deps.createRecallParametersRuntime).not.toHaveBeenCalled();
    expect(socket.handlerCount()).toBe(0);
  });

  it('reports a failed load and retries it on the next event', async () => {
    const socket = createFakeSocketHub();
    const deps = createDeps();
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Error('chunk unavailable'))
      .mockResolvedValueOnce({ createRecallParametersRuntime: deps.createRecallParametersRuntime });
    const bridge = attachRecallParametersRuntime({ ...deps, hub: socket.hub, load });

    socket.emit('recall_parameters_updated', event({ steps: 1 }));
    await flush();

    expect(deps.reportError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'recall-parameters', message: 'chunk unavailable', projectId: 'project-1' })
    );

    socket.emit('recall_parameters_updated', event({ steps: 2 }));
    await flush();

    expect(load).toHaveBeenCalledTimes(2);
    expect(deps.received.map(stepsOf)).toEqual([2]);

    bridge.dispose();
  });
});

describe('attachVideoRecallRuntime', () => {
  it('loads the video runtime on its own event only, handing over the arrival project', async () => {
    const socket = createFakeSocketHub();
    const deps = createDeps();
    const createVideoRecallRuntime = vi.fn(({ replay = [] }: RuntimeOptions) => {
      deps.received.push(...replay);
      return { dispose: deps.dispose };
    });
    const load = vi.fn(() => Promise.resolve({ createVideoRecallRuntime }));
    const getSessionUserId = () => 'owner';
    const bridge = attachVideoRecallRuntime({
      ...deps,
      getSessionUserId,
      hub: socket.hub,
      load,
    });

    socket.emit('recall_parameters_updated', event({ steps: 1 }));
    expect(load).not.toHaveBeenCalled();

    socket.emit('video_recall_requested', { action: 'initial_video', user_id: 'owner' });
    await flush();

    expect(load).toHaveBeenCalledTimes(1);
    expect(createVideoRecallRuntime).toHaveBeenCalledWith(
      expect.objectContaining({ getSessionUserId, reveal: deps.reveal })
    );
    expect(deps.received).toEqual([{ payload: { action: 'initial_video', user_id: 'owner' }, projectId: 'project-1' }]);

    bridge.dispose();
    expect(deps.dispose).toHaveBeenCalledTimes(1);
  });
});
