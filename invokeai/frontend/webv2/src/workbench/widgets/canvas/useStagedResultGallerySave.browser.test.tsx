import type { SaveStagedResultOutcome } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useStagedResultGallerySave } from './useStagedResultGallerySave';

const BOARDS = [
  { archived: false, id: 'board-a', kind: 'board', name: 'Board A' },
  { archived: false, id: 'none', kind: 'uncategorized', name: '' },
];

const ARCHIVED_BOARD = { archived: true, id: 'board-archived', kind: 'board', name: 'Old Board' };

const mocks = vi.hoisted(() => ({
  activeProjectId: 'project-1',
  fetchBoards: vi.fn<(query: { includeArchived?: boolean }) => Promise<unknown>>(),
  findGalleryItem: vi.fn(),
  invalidateGallery: vi.fn(() => Promise.resolve()),
  retry: vi.fn<(...args: unknown[]) => Promise<unknown>>(),
  save: vi.fn<(...args: unknown[]) => Promise<unknown>>(),
}));

vi.mock('@workbench/canvas-operations/api', () => ({
  retryStagedResultBoard: (...args: unknown[]) => mocks.retry(...args),
  saveStagedResultToGallery: (...args: unknown[]) => mocks.save(...args),
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => mocks.findGalleryItem }));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryBoardsOptions: (query: { includeArchived?: boolean }) => ({
    queryFn: () => mocks.fetchBoards(query),
    queryKey: ['test-boards', query.includeArchived === true],
  }),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...(args as [])),
}));
vi.mock('@workbench/WorkbenchContext', () => {
  const project = (id: string) => ({ id, widgetInstances: {} });
  const queries = {
    getSnapshot: () => ({ activeProject: project(mocks.activeProjectId) }),
    isActiveProject: (id: string) => id === mocks.activeProjectId,
  };

  return { useWorkbenchQueries: () => queries };
});

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { retry: 'Retry' },
        widgets: {
          canvas: {
            staging: {
              boardFailed: "Saved to Gallery, but couldn't add it to {{board}}",
              boardFailedDescription: '{{name}} is in the Gallery without a board for now.',
              boardFailedUnnamed: "Saved to Gallery, but couldn't add it to its board",
              boardMissing: 'Saved to Gallery, but its board no longer exists',
              boardMissingDescription: '{{name}} is in Uncategorized.',
              saveError: "Couldn't save to gallery",
              saved: 'Saved to gallery',
              savedDescription: '{{name}} added to your gallery.',
              savedToBoard: 'Saved to {{board}}',
              showInGallery: 'Show in Gallery',
            },
          },
          gallery: { uncategorized: 'Uncategorized' },
        },
      },
    },
  },
});

type Save = ReturnType<typeof useStagedResultGallerySave>;
const saveRef = createRef<Save>();
const Probe = ({ ref }: { ref: Ref<Save> }) => {
  const result = useStagedResultGallerySave('staged.png');

  useImperativeHandle(ref, () => result, [result]);

  return <output data-testid="saving">{String(result.isSaving)}</output>;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
};

const toastRoot = () => document.querySelector<HTMLElement>('[data-part="root"][data-scope="toast"]');
const toastText = () => toastRoot()?.textContent ?? '';
const toastButton = (label: string) =>
  [...(toastRoot()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === label) ?? null;
const until = (assertion: () => void) => act(() => vi.waitFor(assertion, { interval: 20, timeout: 3000 }));

beforeEach(async () => {
  accountLifecycle.activate('staged-save-user');
  mocks.activeProjectId = 'project-1';
  vi.clearAllMocks();
  // The listing hides archived boards, as the Gallery does by default.
  mocks.fetchBoards.mockImplementation(({ includeArchived }) =>
    Promise.resolve(includeArchived ? [...BOARDS, ARCHIVED_BOARD] : BOARDS)
  );
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <Probe ref={saveRef} />
            <AppToaster />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
});

afterEach(async () => {
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
});

const saved = (boardId: string): SaveStagedResultOutcome => ({ boardId, imageName: 'staged.png', status: 'saved' });

describe('saving a staged result to the Gallery', () => {
  it('names the board it went to, refreshes the Gallery, and shows the item on request', async () => {
    mocks.save.mockResolvedValue(saved('board-a'));

    await act(() => saveRef.current!.save('staged.png'));
    await until(() => expect(toastText()).toContain('Saved to Board A'));

    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    await act(() => toastButton('Show in Gallery')!.click());
    // In a Gallery panel only: the Canvas under review keeps the center.
    expect(mocks.findGalleryItem).toHaveBeenCalledExactlyOnceWith(
      { kind: 'image', name: 'staged.png' },
      { revealPreview: false }
    );
  });

  it('names Uncategorized as the Gallery does', async () => {
    mocks.save.mockResolvedValue(saved('none'));

    await act(() => saveRef.current!.save('staged.png'));

    await until(() => expect(toastText()).toContain('Saved to Uncategorized'));
  });

  it('runs one save at a time for an image, and says so while it runs', async () => {
    const pending = deferred<SaveStagedResultOutcome>();
    mocks.save.mockReturnValue(pending.promise);

    let first!: Promise<void>;
    await act(() => {
      first = saveRef.current!.save('staged.png');
    });
    expect(host!.querySelector('[data-testid="saving"]')!.textContent).toBe('true');
    await act(() => saveRef.current!.save('staged.png'));
    expect(mocks.save).toHaveBeenCalledOnce();

    pending.resolve(saved('board-a'));
    await act(() => first);
    expect(host!.querySelector('[data-testid="saving"]')!.textContent).toBe('false');
  });

  it('reports a missed board without claiming success, keeps the item findable, and retries the board', async () => {
    mocks.save.mockResolvedValue({ boardId: 'board-a', imageName: 'staged.png', status: 'board-failed' });
    mocks.retry.mockResolvedValue(saved('board-a'));

    await act(() => saveRef.current!.save('staged.png'));
    await until(() => expect(toastText()).toContain("Saved to Gallery, but couldn't add it to Board A"));
    expect(toastRoot()!.getAttribute('data-type')).toBe('warning');
    expect(mocks.invalidateGallery).toHaveBeenCalledOnce();
    expect(toastButton('Show in Gallery')).not.toBeNull();

    await act(() => toastButton('Retry')!.click());
    expect(mocks.retry).toHaveBeenCalledExactlyOnceWith({ boardId: 'board-a', imageName: 'staged.png' });
    await until(() => expect(toastText()).toContain('Saved to Board A'));
  });

  it('claims nothing when promotion fails', async () => {
    mocks.save.mockRejectedValue(new Error('offline'));

    await act(() => saveRef.current!.save('staged.png'));
    await until(() => expect(toastText()).toContain("Couldn't save to gallery"));

    expect(toastRoot()!.getAttribute('data-type')).toBe('error');
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
    expect(toastButton('Show in Gallery')).toBeNull();
  });

  it('still reports a save that finished after a project switch, but offers no way into the other project', async () => {
    const pending = deferred<SaveStagedResultOutcome>();
    mocks.save.mockReturnValue(pending.promise);

    let saving!: Promise<void>;
    await act(() => {
      saving = saveRef.current!.save('staged.png');
    });
    mocks.activeProjectId = 'project-2';
    pending.resolve(saved('board-a'));
    await act(() => saving);
    await until(() => expect(toastText()).toContain('Saved to Board A'));

    expect(mocks.save.mock.calls[0]?.[0]).toMatchObject({ project: { id: 'project-1' } });
    expect(toastButton('Show in Gallery')).toBeNull();
  });

  it('frees the image for another save once a save fails', async () => {
    mocks.save.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(saved('board-a'));

    await act(() => saveRef.current!.save('staged.png'));
    expect(host!.querySelector('[data-testid="saving"]')!.textContent).toBe('false');
    await act(() => saveRef.current!.save('staged.png'));

    expect(mocks.save).toHaveBeenCalledTimes(2);
    await until(() => expect(toastText()).toContain('Saved to Board A'));
  });

  it('takes its toast away with the account that saved, and does nothing from it meanwhile', async () => {
    mocks.save.mockResolvedValue({ boardId: 'board-a', imageName: 'staged.png', status: 'board-failed' });

    await act(() => saveRef.current!.save('staged.png'));
    await until(() => expect(toastButton('Retry')).not.toBeNull());
    const retry = toastButton('Retry')!;
    const show = toastButton('Show in Gallery')!;
    mocks.invalidateGallery.mockClear();
    await act(() => accountLifecycle.activate('another-user'));

    // The toast names the previous account's image and board: it is leaving, and its buttons act for nobody.
    expect(toastRoot()?.getAttribute('data-state')).toBe('closed');
    await act(() => show.click());
    await act(() => retry.click());

    expect(mocks.findGalleryItem).not.toHaveBeenCalled();
    expect(mocks.retry).not.toHaveBeenCalled();
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
    await until(() => expect(toastRoot()).toBeNull());
    expect(document.body.textContent).not.toContain('Saved to Board A');
  });

  it('takes a plain report away with the account too', async () => {
    const pending = deferred<SaveStagedResultOutcome>();
    mocks.save.mockReturnValue(pending.promise);

    let saving!: Promise<void>;
    await act(() => {
      saving = saveRef.current!.save('staged.png');
    });
    mocks.activeProjectId = 'project-2';
    pending.resolve(saved('board-a'));
    await act(() => saving);
    await until(() => expect(toastText()).toContain('Saved to Board A'));
    expect(toastButton('Show in Gallery')).toBeNull();
    // Checked synchronously around the switch: the toaster's own timeout would also close it within a wait.
    expect(toastRoot()?.getAttribute('data-state')).toBe('open');

    await act(() => accountLifecycle.activate('another-user'));

    expect(toastRoot()?.getAttribute('data-state')).toBe('closed');
    await until(() => expect(toastRoot()).toBeNull());
  });

  it('says a board that no longer exists is gone, without a retry that cannot succeed', async () => {
    mocks.save.mockResolvedValue({ boardId: 'board-deleted', imageName: 'staged.png', status: 'board-failed' });

    await act(() => saveRef.current!.save('staged.png'));
    await until(() => expect(toastText()).toContain('Saved to Gallery, but its board no longer exists'));

    expect(toastText()).toContain('staged.png is in Uncategorized.');
    expect(toastButton('Retry')).toBeNull();
    expect(toastButton('Show in Gallery')).not.toBeNull();
  });

  it('keeps Retry, and the real name, for an archived destination the listing hides', async () => {
    mocks.save.mockResolvedValue({ boardId: 'board-archived', imageName: 'staged.png', status: 'board-failed' });

    await act(() => saveRef.current!.save('staged.png'));

    await until(() => expect(toastText()).toContain("Saved to Gallery, but couldn't add it to Old Board"));
    expect(toastButton('Retry')).not.toBeNull();
    expect(toastText()).not.toContain('no longer exists');
  });

  it('names an archived destination on success too', async () => {
    mocks.save.mockResolvedValue(saved('board-archived'));

    await act(() => saveRef.current!.save('staged.png'));

    await until(() => expect(toastText()).toContain('Saved to Old Board'));
  });

  it('reports without the board name rather than wait on a slow boards listing', async () => {
    mocks.fetchBoards.mockReturnValue(new Promise(() => {}));
    mocks.save.mockResolvedValue(saved('board-a'));

    await act(() => saveRef.current!.save('staged.png'));

    await until(() => expect(toastText()).toContain('Saved to gallery'));
    expect(host!.querySelector('[data-testid="saving"]')!.textContent).toBe('false');
  });
});
