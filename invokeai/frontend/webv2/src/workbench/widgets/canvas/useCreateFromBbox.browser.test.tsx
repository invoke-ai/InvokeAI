import { accountLifecycle } from '@platform/state/accountLifecycle';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useCreateFromBbox } from './useCreateFromBbox';

const mocks = vi.hoisted(() => ({
  createFromBbox: vi.fn<(options: { destination: string }) => Promise<unknown>>(),
  notify: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  reportError: vi.fn(),
}));

vi.mock('@workbench/canvas-operations/api', () => ({
  createFromBbox: (options: { destination: string }) => mocks.createFromBbox(options),
  getCreateFromBboxNotice: () => ({ kind: 'success', titleKey: 'created' }),
}));
vi.mock('@features/models', () => ({
  ensureModelsLoaded: () => Promise.resolve(),
  useModelsSelector: () => [],
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    canvas: { apply: vi.fn() },
    generation: { patchSettings: vi.fn() },
    notifications: { reportError: mocks.reportError },
  }),
  useWorkbenchQueries: () => ({
    getProject: () => undefined,
    getSnapshot: () => ({ activeProject: { id: 'project-1', widgetInstances: {} } }),
    isActiveProject: () => true,
  }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => mocks.notify }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

type Handle = ReturnType<typeof useCreateFromBbox>;
const engine = { projectId: 'project-1' } as unknown as Parameters<typeof useCreateFromBbox>[0];

/** Two probes stand in for the header's Frame menu and the canvas context menu, which each call the hook. */
const Probe = ({ ref }: { ref: Ref<Handle> }) => {
  const handle = useCreateFromBbox(engine);
  useImperativeHandle(ref, () => handle, [handle]);
  return null;
};

const deferred = () => {
  let resolve!: (value: unknown) => void;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const header = createRef<Handle>();
const menu = createRef<Handle>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = (hosts: { header: boolean; menu: boolean }) =>
  act(() =>
    root?.render(
      <>
        {hosts.header ? <Probe ref={header} /> : null}
        {hosts.menu ? <Probe ref={menu} /> : null}
      </>
    )
  );

beforeEach(async () => {
  vi.clearAllMocks();
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await render({ header: true, menu: true });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useCreateFromBbox across surfaces', () => {
  it('runs one create per project at a time, busy on every surface, and releases it once settled', async () => {
    const inFlight = deferred();
    mocks.createFromBbox.mockReturnValueOnce(inFlight.promise);

    let first: Promise<void> | undefined;
    await act(() => {
      first = header.current!.createFromBbox('raster');
    });
    expect(header.current!.isCreating).toBe(true);
    expect(menu.current!.isCreating).toBe(true);

    await act(() => menu.current!.createFromBbox('control'));
    expect(mocks.createFromBbox).toHaveBeenCalledOnce();

    await act(async () => {
      inFlight.resolve({ status: 'created' });
      await first;
    });
    expect(header.current!.isCreating).toBe(false);
    expect(menu.current!.isCreating).toBe(false);
  });

  it('releases the gate after a failed create', async () => {
    mocks.createFromBbox.mockRejectedValueOnce(new Error('export failed'));

    await act(() => header.current!.createFromBbox('raster'));
    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(menu.current!.isCreating).toBe(false);

    mocks.createFromBbox.mockResolvedValueOnce({ status: 'created' });
    await act(() => menu.current!.createFromBbox('raster'));
    expect(mocks.createFromBbox).toHaveBeenCalledTimes(2);
  });

  it('releases the gate when the surface that started the create unmounts before it settles', async () => {
    const inFlight = deferred();
    mocks.createFromBbox.mockReturnValueOnce(inFlight.promise);

    let first: Promise<void> | undefined;
    await act(() => {
      first = header.current!.createFromBbox('raster');
    });
    await render({ header: false, menu: true });
    expect(menu.current!.isCreating).toBe(true);

    await act(async () => {
      inFlight.resolve({ status: 'created' });
      await first;
    });
    expect(menu.current!.isCreating).toBe(false);
  });
});
