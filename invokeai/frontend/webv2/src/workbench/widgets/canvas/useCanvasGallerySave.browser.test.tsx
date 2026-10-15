import type { CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useCanvasGallerySave } from './useCanvasGallerySave';

const mocks = vi.hoisted(() => ({
  reportError: vi.fn(),
  saveCanvasToGallery: vi.fn<(options: { region: CanvasGallerySaveRegion }) => Promise<unknown>>(),
}));

vi.mock('@workbench/canvas-operations/api', () => ({
  saveCanvasToGallery: (...args: [{ region: CanvasGallerySaveRegion }]) => mocks.saveCanvasToGallery(...args),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ notifications: { reportError: mocks.reportError } }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: { id: 'project-1' } }) }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));
vi.mock('@features/gallery/queries', () => ({ invalidateGallery: vi.fn(() => Promise.resolve()) }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

type Handle = ReturnType<typeof useCanvasGallerySave>;
const engine = { projectId: 'project-1' } as unknown as Parameters<typeof useCanvasGallerySave>[0];

/** Two probes stand in for the header button and the context menu, which each call the hook. */
const Probe = ({ ref }: { ref: Ref<Handle> }) => {
  const handle = useCanvasGallerySave(engine);
  useImperativeHandle(ref, () => handle, [handle]);
  return null;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const header = createRef<Handle>();
const menu = createRef<Handle>();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const deferred = () => {
  let resolve!: (value: unknown) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, reject, resolve };
};

beforeEach(async () => {
  vi.clearAllMocks();
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <Probe ref={header} />
        <Probe ref={menu} />
      </QueryClientProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useCanvasGallerySave across surfaces', () => {
  it('runs one save per project at a time and shows it busy on every surface', async () => {
    const inFlight = deferred();
    mocks.saveCanvasToGallery.mockReturnValueOnce(inFlight.promise);

    let first: Promise<void> | undefined;
    await act(() => {
      first = header.current!.save('canvas');
    });
    expect(header.current!.isSaving).toBe(true);
    expect(menu.current!.isSaving).toBe(true);

    await act(() => menu.current!.save('bbox'));
    expect(mocks.saveCanvasToGallery).toHaveBeenCalledTimes(1);

    await act(async () => {
      inFlight.resolve({ imageName: 'saved.png', status: 'saved' });
      await first;
    });
    expect(header.current!.isSaving).toBe(false);
    expect(menu.current!.isSaving).toBe(false);
  });

  it('releases the gate after a failed save', async () => {
    mocks.saveCanvasToGallery.mockRejectedValueOnce(new Error('upload failed'));
    await act(() => header.current!.save('canvas'));
    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(menu.current!.isSaving).toBe(false);

    mocks.saveCanvasToGallery.mockResolvedValueOnce({ imageName: 'saved.png', status: 'saved' });
    await act(() => menu.current!.save('canvas'));
    expect(mocks.saveCanvasToGallery).toHaveBeenCalledTimes(2);
  });
});
