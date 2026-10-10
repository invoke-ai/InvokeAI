import type { GalleryBoard } from '@features/gallery';
import type * as transportModule from '@platform/transport/http';

import { ChakraProvider } from '@chakra-ui/react';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { apiFetchJson } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { DeleteProjectDialog } from './DeleteProjectDialog';

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof transportModule>()),
  apiFetchJson: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const messages: Record<string, string> = {
        'projects.deleteProject': 'Delete project',
        'projects.deleteProjectBoardsUnknown': 'Its other boards could not be checked.',
        'projects.deleteProjectCheckingBoards': 'Checking its boards…',
        'projects.deleteProjectDeleteBoards': 'Delete its other boards too',
        'projects.deleteProjectDeleteBoardsHint': `${String(values?.items)} return to Uncategorized.`,
        'projects.deleteProjectInboxNote': 'Its inbox goes with it.',
        'projects.deleteProjectItemCount': `${String(values?.count)} items`,
        'projects.deleteProjectKeepBoards': 'Keep its other boards in the Library',
        'projects.deleteProjectKeepBoardsHint': `${String(values?.boards)} stay where you can find them.`,
        'projects.deleteProjectQuestion': 'Delete project?',
      };

      return messages[key] ?? key;
    },
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  projectId: null,
  videoCount: 0,
  ...overrides,
});

const boards = [
  createBoard({ id: 'none', kind: 'uncategorized', name: '' }),
  createBoard({ id: 'inbox', imageCount: 142, isInbox: true, name: 'Mahogany House', projectId: 'p1' }),
  createBoard({ assetCount: 2, id: 'facades', imageCount: 37, name: 'Façade variants', projectId: 'p1' }),
  createBoard({ id: 'site', name: 'Site plan refs', projectId: 'p1', videoCount: 1 }),
  createBoard({ archived: true, id: 'old', imageCount: 5, name: 'Old façades', projectId: 'p1' }),
  createBoard({ id: 'theirs', isInbox: true, name: 'Harbor Tower', projectId: 'p2' }),
];

const onConfirm = vi.fn();
const noop = () => undefined;
let host: HTMLDivElement | null = null;
let root: Root | null = null;

const renderDialog = async (projectId: string, seed = true) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  if (seed) {
    // Archived members are included: they go with the project just the same.
    queryClient.setQueryData(galleryBoardsOptions({ includeArchived: true }).queryKey, boards);
  }

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <DeleteProjectDialog
            body="Delete this project?"
            isOpen
            projectId={projectId}
            onClose={noop}
            onConfirm={onConfirm}
          />
        </QueryClientProvider>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const confirmButton = () =>
  [...document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')].find(
    (button) => button.textContent === 'Delete project'
  )!;

beforeEach(() => {
  onConfirm.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('DeleteProjectDialog', () => {
  it('names the other boards and what they hold, and keeps them by default', async () => {
    await renderDialog('p1');

    const text = document.querySelector('[role="alertdialog"]')?.textContent ?? '';
    expect(text).toContain('Façade variants, Site plan refs, Old façades stay where you can find them.');
    expect(text).toContain('45 items return to Uncategorized.');
    expect(text).toContain('Its inbox goes with it.');

    await act(() => userEvent.click(confirmButton()));

    expect(onConfirm).toHaveBeenCalledExactlyOnceWith('release');
  });

  it('deletes the other boards only when that is chosen', async () => {
    await renderDialog('p1');

    const deleteOption = [...document.querySelectorAll<HTMLElement>('[role="alertdialog"] label')].find((label) =>
      label.textContent?.includes('Delete its other boards too')
    )!;
    await act(() => userEvent.click(deleteOption));
    await act(() => userEvent.click(confirmButton()));

    expect(onConfirm).toHaveBeenCalledExactlyOnceWith('delete');
  });

  it('holds confirmation until the boards are known, and says so when they cannot be', async () => {
    vi.mocked(apiFetchJson).mockReturnValue(new Promise(() => {}));
    await renderDialog('p1', false);

    expect(confirmButton().disabled).toBe(true);
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Checking its boards…');

    vi.mocked(apiFetchJson).mockRejectedValue(new Error('offline'));
    await act(() => root?.unmount());
    root = createRoot(host!);
    await renderDialog('p1', false);

    await vi.waitFor(() => {
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain(
        'Its other boards could not be checked.'
      );
    });
    expect(confirmButton().disabled).toBe(false);
    expect(document.querySelector('[role="alertdialog"]')?.textContent).not.toContain('Delete its other boards too');

    await act(() => userEvent.click(confirmButton()));
    expect(onConfirm).toHaveBeenCalledExactlyOnceWith('release');
  });

  it('asks nothing about boards when the project holds only its inbox', async () => {
    await renderDialog('p2');

    const text = document.querySelector('[role="alertdialog"]')?.textContent ?? '';
    expect(text).not.toContain('Keep its other boards');
    expect(text).toContain('Its inbox goes with it.');
  });
});
