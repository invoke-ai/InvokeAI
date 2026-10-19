/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { ImageIndexAvailability } from '@features/gallery/data/backend';
import type { GalleryUiAdapter } from '@features/gallery/react';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryUiProvider } from '@features/gallery/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, useMemo, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryItemSearch, SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS } from './GalleryItemSearch';
import { GalleryWidgetContext } from './GalleryWidgetContext';

const mocks = vi.hoisted(() => ({
  fetchImageIndexAvailability: vi.fn<() => Promise<ImageIndexAvailability>>(),
}));

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  fetchImageIndexAvailability: () => mocks.fetchImageIndexAvailability(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, options?: Record<string, string>) => (options?.model ? `${key}:${options.model}` : key),
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const notificationsAdd = vi.fn();
const noop = vi.fn();
const NoopProvider = ({ children }: { children: ReactNode }) => children;
const adapter: GalleryUiAdapter = {
  ItemActionsProvider: NoopProvider,
  ImageContextMenu: () => null,
  antialiasProgressImages: false,
  exportProject: noop,
  gallery: {
    clearSelection: noop,
    reconcileDeletedBoardOutcome: noop,
    selectBoard: noop,
    selectImage: noop,
    selectItem: noop,
    setCompareImage: noop,
    setCompareItem: noop,
    setItemMultiSelection: noop,
    setPage: noop,
    setPageInfo: noop,
    setSearchTerm: noop,
    setStarredOnly: noop,
    setSemanticSearchMode: noop,
    setSemanticSearchText: noop,
    commitSemanticSearch: noop,
    clearSearch: noop,
    setView: noop,
    toggleItemSelection: noop,
    updateSettings: noop,
  },
  galleryValues: {},
  generateValues: {},
  getItemLabel: () => Promise.resolve(null),
  liveFollowEnabled: false,
  progressSessions: [],
  pinnedProgressSessionId: null,
  followedProgressSessionId: null,
  followProgressSession: noop,
  notifications: { add: notificationsAdd, reportError: noop },
  projectId: 'project-1',
  projectName: 'Project',
  projects: [],
  widgets: { openGallery: () => true, patchGalleryValues: noop },
};

const actions = {
  clearSearch: vi.fn(),
  commitSemanticSearch: vi.fn(),
  setSearchTerm: vi.fn(),
  setSemanticImageQuery: vi.fn(),
  setSemanticSearchMode: vi.fn(),
  setSemanticSearchText: vi.fn(),
};

const archivedBoards = [{ archived: true, id: 'old-board' }];

interface HarnessGallery {
  boards: Array<Pick<GalleryBoard, 'archived' | 'id'>>;
  searchTerm: string;
  selectedBoardId: string;
  semanticImageQuery: GallerySemanticReference | null;
  semanticSearchText: string | null;
}

/** Write text intents back into controlled values to model the workbench while recording calls. */
const Harness = ({ initial }: { initial: Partial<HarnessGallery> }) => {
  const [gallery, setGallery] = useState<HarnessGallery>({
    boards: [],
    searchTerm: '',
    selectedBoardId: 'none',
    semanticImageQuery: null,
    semanticSearchText: null,
    ...initial,
  });
  const contextValue = useMemo(
    () =>
      ({
        actions: {
          ...actions,
          setSearchTerm: (searchTerm: string) => {
            actions.setSearchTerm(searchTerm);
            setGallery((current) => ({ ...current, searchTerm }));
          },
          setSemanticSearchText: (semanticSearchText: string) => {
            actions.setSemanticSearchText(semanticSearchText);
            setGallery((current) => ({ ...current, semanticSearchText }));
          },
        },
        gallery,
      }) as unknown as GalleryWidgetContextValue,
    [gallery]
  );

  return (
    <GalleryWidgetContext value={contextValue}>
      <GalleryItemSearch />
    </GalleryWidgetContext>
  );
};

const renderSearch = async (
  gallery: Partial<HarnessGallery> = {},
  availability: ImageIndexAvailability | 'unresolved' = { modelName: null, state: 'ready' }
) => {
  if (availability === 'unresolved') {
    mocks.fetchImageIndexAvailability.mockReturnValue(
      new Promise(() => {
        // Never settles: the field renders before availability is known.
      })
    );
  } else {
    mocks.fetchImageIndexAvailability.mockResolvedValue(availability);
  }

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient ?? new QueryClient()}>
          <GalleryUiProvider adapter={adapter}>
            <Harness initial={gallery} />
          </GalleryUiProvider>
        </QueryClientProvider>
      </ChakraProvider>
    );
  });

  if (availability !== 'unresolved') {
    // The availability query settles after render; the toggle follows it.
    await vi.waitFor(() => expect(queryClient?.getQueryCache().findAll()[0]?.state.status).toBe('success'));
    await act(async () => {
      await Promise.resolve();
    });
  }
};

const getToggle = () =>
  host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.semanticSearchAriaLabel"]') ?? null;

const getInput = (): HTMLInputElement => {
  const input = host?.querySelector('input');

  if (!input) {
    throw new Error('search input did not render');
  }

  return input;
};

const typeInto = async (value: string) => {
  await page.getByRole('textbox').fill(value);
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'left:0;position:fixed;top:0;width:320px;';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  Object.values(actions).forEach((action) => action.mockClear());
  notificationsAdd.mockClear();
  mocks.fetchImageIndexAvailability.mockReset();
});

afterEach(async () => {
  vi.useRealTimers();
  await act(() => root?.unmount());
  document.querySelectorAll('[data-scope="popover"][data-part="positioner"]').forEach((element) => element.remove());
  queryClient?.clear();
  queryClient = null;
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryItemSearch help', () => {
  it('shows the relative value in a valid prefixed token', async () => {
    await renderSearch();
    const trigger = host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.searchHelpTitle"]');

    await act(async () => {
      trigger?.click();
      await Promise.resolve();
    });

    const examples = Array.from(document.querySelectorAll('code')).map((element) => element.textContent);

    expect(examples).toContain('from:7d');
    expect(examples).not.toContain('7d');
  });
});

describe('GalleryItemSearch semantic mode', () => {
  it('offers no toggle when indexing is not configured', async () => {
    await renderSearch({}, { modelName: null, state: 'disabled' });

    expect(getToggle()).toBeNull();
  });

  it('offers the toggle when indexing is configured, even without the model installed', async () => {
    await renderSearch({}, { modelName: 'clip', state: 'model_missing' });

    expect(getToggle()).not.toBeNull();
    expect(getToggle()?.getAttribute('aria-pressed')).toBe('false');
  });

  it('keeps the toggle reachable for a field already in semantic mode while availability is unknown', async () => {
    await renderSearch({ semanticSearchText: 'sunset' }, 'unresolved');

    expect(getToggle()?.getAttribute('aria-pressed')).toBe('true');
  });

  it('enters semantic mode from the toggle and hands the field focus', async () => {
    await renderSearch();

    await act(() => getToggle()?.click());

    expect(actions.setSemanticSearchMode).toHaveBeenCalledWith(true);
    expect(document.activeElement).toBe(getInput());
    expect(notificationsAdd).not.toHaveBeenCalled();
  });

  it('tells the user which model to install when the toggle is pressed without one', async () => {
    await renderSearch({}, { modelName: 'DFN2B-CLIP', state: 'model_missing' });

    await act(() => getToggle()?.click());

    expect(actions.setSemanticSearchMode).toHaveBeenCalledWith(true);
    expect(notificationsAdd).toHaveBeenCalledWith({
      kind: 'info',
      message: 'widgets.gallery.semanticSearchInstallModel:DFN2B-CLIP',
      title: 'widgets.gallery.semanticSearchModelMissingTitle',
    });
  });

  it('marks the field, keeps the typed text live, and ranks it only after a typing pause', async () => {
    await renderSearch({ semanticSearchText: '' });
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

    expect(getInput().closest('[data-mode]')?.getAttribute('data-mode')).toBe('semantic');
    expect(getInput().getAttribute('aria-label')).toBe('widgets.gallery.semanticSearchAriaLabel');
    expect(getToggle()?.getAttribute('aria-pressed')).toBe('true');
    // The metadata grammar help does not apply to a description.
    expect(host?.querySelector('button[aria-label="widgets.gallery.searchHelpTitle"]')).toBeNull();

    await typeInto('su');
    await typeInto('sun');

    expect(actions.setSemanticSearchText.mock.calls).toEqual([['su'], ['sun']]);
    expect(actions.setSearchTerm).not.toHaveBeenCalled();
    expect(actions.commitSemanticSearch).not.toHaveBeenCalled();

    await act(() => vi.advanceTimersByTime(SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS - 1));

    expect(actions.commitSemanticSearch).not.toHaveBeenCalled();

    await act(() => vi.advanceTimersByTime(1));

    // One commit, for the text the pause ended on.
    expect(actions.commitSemanticSearch.mock.calls).toEqual([['sun']]);
  });

  it('commits at once on Enter and drops the pending timer', async () => {
    await renderSearch({ semanticSearchText: 'sun' });
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

    await typeInto('sunset');

    expect(getInput().value).toBe('sunset');

    await act(() => getInput().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' })));

    expect(actions.commitSemanticSearch.mock.calls).toEqual([['sunset']]);

    await act(() => vi.advanceTimersByTime(SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS));

    expect(actions.commitSemanticSearch).toHaveBeenCalledOnce();
  });

  it('leaves semantic mode from the toggle and clears everything from the clear button', async () => {
    await renderSearch({ semanticSearchText: 'sunset' });

    await act(() => getToggle()?.click());

    expect(actions.setSemanticSearchMode).toHaveBeenCalledWith(false);

    await act(() =>
      host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.clearSemanticSearch"]')?.click()
    );

    expect(actions.clearSearch).toHaveBeenCalledOnce();
  });

  it('explains a missing model under the field, and still ranks the text so nothing is left to reconcile', async () => {
    await renderSearch({ semanticSearchText: '' }, { modelName: 'DFN2B-CLIP', state: 'model_missing' });
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

    const hint = host?.querySelector('[role="status"]');

    expect(hint?.textContent).toBe('widgets.gallery.semanticSearchModelMissing:DFN2B-CLIP');
    expect(getInput().getAttribute('aria-describedby')).toBe(hint?.id);

    await typeInto('sun');
    await act(() => vi.advanceTimersByTime(SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS));

    expect(actions.commitSemanticSearch.mock.calls).toEqual([['sun']]);
  });

  it('explains why a semantic search in an archived board finds nothing', async () => {
    await renderSearch({ boards: archivedBoards, selectedBoardId: 'old-board', semanticSearchText: 'sun' });

    expect(host?.querySelector('[role="status"]')?.textContent).toBe('widgets.gallery.semanticSearchArchivedBoard');
  });

  it('explains why an image similarity search in an archived board finds nothing', async () => {
    await renderSearch({
      boards: archivedBoards,
      selectedBoardId: 'old-board',
      semanticImageQuery: { imageName: 'ref.png', kind: 'image' },
    });

    expect(host?.querySelector('[role="status"]')?.textContent).toBe('widgets.gallery.semanticSearchArchivedBoard');
  });

  it('leaves a cluster in an archived board unexplained, since it lists its members', async () => {
    await renderSearch({
      boards: archivedBoards,
      selectedBoardId: 'old-board',
      semanticImageQuery: { clusterId: 'c1', kind: 'cluster', label: 'Boats' },
    });

    expect(host?.querySelector('[role="status"]')).toBeNull();
  });

  it('leaves a metadata search in an archived board unexplained, since it still matches', async () => {
    await renderSearch({ boards: archivedBoards, searchTerm: 'sun', selectedBoardId: 'old-board' });

    expect(host?.querySelector('[role="status"]')).toBeNull();
  });

  it('shows no hint on a ready index', async () => {
    await renderSearch({ semanticSearchText: 'sun' });

    expect(host?.querySelector('[role="status"]')).toBeNull();
  });

  it('renders a reference as a chip with the toggle beside it, and a text ranking as the field', async () => {
    await renderSearch({ semanticImageQuery: { imageName: 'ref.png', kind: 'image' } });

    expect(host?.querySelector('input')).toBeNull();
    expect(host?.textContent).toContain('widgets.gallery.semanticSimilarTo');
    expect(getToggle()?.getAttribute('aria-pressed')).toBe('false');

    await act(() => getToggle()?.click());

    expect(actions.setSemanticSearchMode).toHaveBeenCalledWith(true);
  });

  it('renders a text ranking saved before the mode existed as a chip', async () => {
    await renderSearch({ semanticImageQuery: { kind: 'text', query: 'old' }, semanticSearchText: null });

    expect(host?.querySelector('input')).toBeNull();
    expect(host?.textContent).toContain('widgets.gallery.semanticTextSearch');
  });

  it('renders a text ranking in semantic mode as the field, not a chip', async () => {
    await renderSearch({ semanticImageQuery: { kind: 'text', query: 'sun' }, semanticSearchText: 'sun' });

    expect(getInput().value).toBe('sun');
    expect(host?.textContent).not.toContain('widgets.gallery.semanticTextSearch');
  });

  it('drops a pending commit when the mode is left', async () => {
    await renderSearch({ semanticSearchText: '' });
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });

    await typeInto('sun');
    await act(() => getToggle()?.click());
    await act(() => vi.advanceTimersByTime(SEMANTIC_SEARCH_COMMIT_DEBOUNCE_MS));

    expect(actions.setSemanticSearchMode).toHaveBeenCalledWith(false);
    expect(actions.commitSemanticSearch).not.toHaveBeenCalled();
  });

  it('searches metadata as before outside semantic mode', async () => {
    await renderSearch({ searchTerm: 'cat' });

    expect(getInput().closest('[data-mode]')?.getAttribute('data-mode')).toBe('metadata');

    await typeInto('cats');

    expect(actions.setSearchTerm).toHaveBeenCalledWith('cats');
    expect(actions.setSemanticSearchText).not.toHaveBeenCalled();

    await act(() => host?.querySelector<HTMLButtonElement>('button[aria-label="common.clearSearch"]')?.click());

    expect(actions.clearSearch).toHaveBeenCalledOnce();
  });
});
