import type { CanvasGallerySaveRegion } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { invalidateGallery, invalidateGalleryBoardDestinations } from '@features/gallery/queries';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { AppToaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, createRef, type Ref, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useCanvasGallerySave } from './useCanvasGallerySave';

const mocks = vi.hoisted(() => ({
  activeProjectId: 'project-1',
  findGalleryItem: vi.fn(),
  reportError: vi.fn(),
  saveCanvasToGallery: vi.fn<(options: { region: CanvasGallerySaveRegion }) => Promise<unknown>>(),
}));

vi.mock('@workbench/canvas-operations/api', () => ({
  saveCanvasToGallery: (...args: [{ region: CanvasGallerySaveRegion }]) => mocks.saveCanvasToGallery(...args),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ notifications: { reportError: mocks.reportError } }),
  useWorkbenchQueries: () => ({
    getSnapshot: () => ({ activeProject: { id: 'project-1', widgetInstances: {} } }),
    isActiveProject: (id: string) => id === mocks.activeProjectId,
  }),
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => mocks.findGalleryItem }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // The auto-add board, B, as the Gallery's board listing names it.
  galleryBoardsOptions: (query: { includeArchived?: boolean }) => ({
    queryFn: () => Promise.resolve([{ archived: false, id: 'board-b', kind: 'board', name: 'Board B' }]),
    queryKey: ['test-boards', query.includeArchived === true],
  }),
  invalidateGallery: vi.fn(() => Promise.resolve()),
  invalidateGalleryBoardDestinations: vi.fn(() => Promise.resolve()),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { board?: string }) => (options?.board ? `${key}: ${options.board}` : key),
  }),
}));

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
  mocks.activeProjectId = 'project-1';
  accountLifecycle.activate('user-a');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={new QueryClient()}>
          <Probe ref={header} />
          <Probe ref={menu} />
          <AppToaster />
        </QueryClientProvider>
      </ChakraProvider>
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
      inFlight.resolve({ boardId: 'board-b', imageName: 'saved.png', status: 'saved' });
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

    mocks.saveCanvasToGallery.mockResolvedValueOnce({ boardId: 'board-b', imageName: 'saved.png', status: 'saved' });
    await act(() => menu.current!.save('canvas'));
    expect(mocks.saveCanvasToGallery).toHaveBeenCalledTimes(2);
  });
  it.each([
    [
      'refuses the board (403)',
      new ApiError(JSON.stringify({ detail: 'Not authorized to change this board' }), 403),
      1,
    ],
    ['no longer has the board (404)', new ApiError(JSON.stringify({ detail: 'Board not found' }), 404), 1],
    ['fails otherwise', new Error('upload failed'), 0],
  ])('reads the board listing again only when the server %s', async (_case, error, refreshes) => {
    mocks.saveCanvasToGallery.mockRejectedValueOnce(error);

    await act(() => header.current!.save('canvas'));

    expect(mocks.reportError).toHaveBeenCalledOnce();
    expect(invalidateGalleryBoardDestinations).toHaveBeenCalledTimes(refreshes);
    expect(invalidateGallery).not.toHaveBeenCalled();
  });
});

const toastRoot = () => document.querySelector<HTMLElement>('[data-part="root"][data-scope="toast"]');
const toastButton = (label: string) =>
  [...(toastRoot()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === label) ?? null;
const until = (assertion: () => void) => act(() => vi.waitFor(assertion, { interval: 20, timeout: 3000 }));

describe('useCanvasGallerySave feedback', () => {
  it('names the board the image went to and shows it in a Gallery panel, leaving the center alone', async () => {
    mocks.saveCanvasToGallery.mockResolvedValueOnce({ boardId: 'board-b', imageName: 'saved.png', status: 'saved' });

    await act(() => header.current!.save('canvas'));

    await until(() => expect(toastRoot()?.textContent).toContain('widgets.canvas.staging.savedToBoard: Board B'));
    await act(() => toastButton('widgets.canvas.staging.showInGallery')!.click());
    expect(mocks.findGalleryItem).toHaveBeenCalledWith({ kind: 'image', name: 'saved.png' }, { revealPreview: false });
  });

  it('offers no Gallery action once another project is active', async () => {
    mocks.saveCanvasToGallery.mockImplementationOnce(() => {
      mocks.activeProjectId = 'project-2';
      return Promise.resolve({ boardId: 'board-b', imageName: 'saved.png', status: 'saved' });
    });

    await act(() => header.current!.save('canvas'));

    await until(() => expect(toastRoot()?.textContent).toContain('Board B'));
    expect(toastButton('widgets.canvas.staging.showInGallery')).toBeNull();
  });
});
