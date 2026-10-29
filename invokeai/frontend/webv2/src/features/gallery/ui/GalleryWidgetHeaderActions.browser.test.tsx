/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';
import type { GalleryUiAdapter } from '@features/gallery/react';

import { ChakraProvider } from '@chakra-ui/react';
import { DEFAULT_GALLERY_SETTINGS, getGallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { GalleryUiProvider } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GalleryWidgetHeaderActions } from './GalleryWidgetChrome';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values === undefined
        ? key
        : `${key}(${Object.entries(values)
            .map(([name, value]) => `${name}=${String(value)}`)
            .join(',')})`,
  }),
}));

const board = (id: string, overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  access: { actions: ['read', 'add_media'], adminOverride: false, generalAccess: 'restricted', role: 'owner' },
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  name: id,
  projectId: null,
  videoCount: 0,
  ...overrides,
});

const BOARDS = [
  board('inbox', { isInbox: true, name: 'Project', projectId: 'p1' }),
  board('concepts', { projectId: 'p1' }),
  board('elsewhere', { projectId: 'p2' }),
  board('library-board'),
  board('view-only', {
    access: { actions: ['read'], adminOverride: false, generalAccess: 'viewer', role: 'viewer' },
    ownerName: 'Dee',
  }),
];

const updateSettings = vi.fn();
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (values: Record<string, unknown>) => {
  const settings = getGallerySettings(values);
  queryClient.setQueryData(
    galleryBoardsOptions({
      includeArchived: settings.showArchivedBoards,
      includeDateBoards: settings.showDateBoards,
      orderBy: settings.boardOrderBy,
      orderDir: settings.boardOrderDir,
    }).queryKey,
    BOARDS
  );
  const adapter = {
    gallery: { updateSettings },
    galleryValues: { projectBoardId: 'inbox', selectedBoardId: 'concepts', ...values },
    projectId: 'p1',
    projectName: 'Project',
    projects: [{ id: 'p2', name: 'Other' }],
  } as unknown as GalleryUiAdapter;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <GalleryUiProvider adapter={adapter}>
            <GalleryWidgetHeaderActions />
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

const trigger = () => document.querySelector<HTMLButtonElement>('[data-gallery-destination]')!;
const openMenu = async () => {
  await act(() => userEvent.click(trigger()));
  await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());
};
const radio = (value: string) => document.querySelector<HTMLElement>(`[role="menuitemradio"][data-value="${value}"]`);

beforeEach(() => {
  accountLifecycle.activate('header-actions-test');
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  updateSettings.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  accountLifecycle.invalidate();
});

describe('GalleryWidgetHeaderActions', () => {
  it('names the project inbox when no destination is chosen, whatever board is browsed', async () => {
    await render({});

    expect(trigger().dataset.galleryDestination).toBe('inbox');
    expect(trigger().getAttribute('aria-label')).toBe('widgets.gallery.destinationTrigger(name=widgets.gallery.inbox)');
    // Closed, the menu renders nothing.
    expect(document.querySelector('[role="menuitemradio"]')).toBeNull();
  });

  it("lists the project's boards and the Library, checks the default, and stores choosing it as no choice", async () => {
    await render({ autoAddBoardId: 'concepts' });
    await openMenu();

    expect(radio('concepts')?.getAttribute('aria-checked')).toBe('true');
    expect(radio('elsewhere')).toBeNull();
    expect(radio('library-board')).not.toBeNull();
    expect(radio('none')).not.toBeNull();

    await act(() => userEvent.click(radio('default')!));

    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: null, followSelectedBoard: false });
  });

  it('turns following the selected board on from the menu', async () => {
    await render({});
    await openMenu();

    await act(() =>
      userEvent.click(
        document.querySelector<HTMLElement>('[role="menuitemcheckbox"][data-value="follow-selected-board"]')!
      )
    );

    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ followSelectedBoard: true });
  });

  it('names the inbox, and the board that refused them, when the chosen board cannot take results', async () => {
    await render({ autoAddBoardId: 'view-only' });

    expect(trigger().dataset.galleryDestination).toBe('inbox');
    await act(() => userEvent.hover(trigger()));
    await vi.waitFor(() =>
      expect(document.body.textContent).toContain(
        'widgets.gallery.destinationRedirectedTooltip(destination=widgets.gallery.inbox,name=view-only)'
      )
    );
  });

  it('follows the selected board while that is on', async () => {
    await render({ followSelectedBoard: true });

    expect(trigger().dataset.galleryDestination).toBe('concepts');
    expect(DEFAULT_GALLERY_SETTINGS.followSelectedBoard).toBe(false);
  });
});
