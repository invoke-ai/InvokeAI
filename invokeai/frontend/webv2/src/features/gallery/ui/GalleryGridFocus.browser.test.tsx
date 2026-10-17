/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { QueueProgressSession } from '@features/queue/contracts';
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type * as projectsApi from '@workbench/projects/api';
import type { WidgetContributionSource } from '@workbench/widgetContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { toGalleryItemKey, toGalleryItemRef } from '@features/gallery/core/items';
import { getGalleryDeletionSuccessor } from '@features/gallery/core/selection';
import { getGallerySettings, type GallerySettings } from '@features/gallery/core/settings';
import { GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import {
  useDeletionConfirmation,
  type RequestDeletionConfirmation,
} from '@workbench/image-actions/useDeletionConfirmation';
import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from '@workbench/settings/store';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, useMemo, useSyncExternalStore, type ReactNode, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryActions, GalleryWidgetContextValue } from './GalleryWidgetContext';
import type { GalleryListingState } from './useGalleryData';

import { mergeGalleryLoadedItems } from './galleryGridLayout';
import { GalleryImageGrid } from './GalleryImageGrid';
import { GalleryWidgetContext } from './GalleryWidgetContext';

/**
 * Keyboard focus in the real grid: the real virtualizer and real key presses, routed to the gallery's commands by
 * the workbench hotkey runtime through real focus regions. The gallery's slice of workbench state is a small
 * external store, and item deletion stands in for the backend round trip that decides the next selection.
 */

const runtimeMocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) =>
    selector(runtimeMocks.store.getSnapshot().activeProject),
  useOptionalWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchInternalStore: () => runtimeMocks.store,
  useWorkbenchQueries: () => runtimeMocks.store.queries,
  useWorkbenchSubscription: () => runtimeMocks.store.subscribe,
}));
// They need the whole application; the gallery registers its own commands.
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));
vi.mock('@features/gallery/data/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // The listing's order, which Shift ranges resolve against.
  galleryItemNamesOptions: () => ({
    queryFn: () => ({ items: state.gallery.items.map(toGalleryItemRef) }),
    queryKey: ['test-gallery-item-names', state.gallery.selectedBoardId],
  }),
  imageIndexAvailabilityOptions: () => ({
    queryFn: () => ({ modelName: null, state: 'disabled' }),
    queryKey: ['test-image-index-availability'],
  }),
}));
// Hotkey preferences save as Settings saves them; the backend round trip is not under test.
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));
vi.mock('@features/queue/react', () => ({
  useItemProgress: () => null,
  useQueueItemProgressImage: () => null,
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: {
            collapseStarredItems: 'Collapse starred items',
            deleteConfirmLabel: 'Delete',
            deleteImagesConfirmTitle: 'Delete image?',
            inProgress: 'In progress',
            contentAriaLabel: 'Gallery content',
            itemsAriaLabel: 'Gallery items',
            loadingBackendGallery: 'Loading gallery',
            selectImageForPreview: 'Select {{name}} for preview',
            starImage: 'Star {{name}}',
            starredItems: 'Starred',
            unstarImage: 'Unstar {{name}}',
          },
        },
      },
    },
  },
});

const createItem = (name: string, overrides: Partial<GalleryItem> = {}): GalleryItem =>
  ({
    boardId: 'board-a',
    category: 'general',
    createdAt: '2026-07-30T00:00:00.000Z',
    fullUrl: `/full/${name}`,
    height: 96,
    isIntermediate: false,
    kind: 'image',
    name,
    starred: false,
    thumbnailUrl: `/thumbnail/${name}`,
    width: 128,
    ...overrides,
  }) as GalleryItem;

const createItems = (count: number, prefix = 'image') =>
  Array.from({ length: count }, (_, index) => createItem(`${prefix}-${index}.png`));

const createSession = (id: string, state: QueueProgressSession['state']): QueueProgressSession => ({
  backendItemId: 10,
  height: 768,
  id,
  itemCount: 1,
  itemIndex: 1,
  label: `Run ${id}`,
  queueItemId: id,
  sourceId: 'generate',
  state,
  width: 512,
});

const READY_LISTING: GalleryListingState = {
  error: null,
  isFetchingMore: false,
  isRetrying: false,
  retry: () => Promise.resolve(),
  status: 'ready',
};

const selectionOf = (item: GalleryItem | null) => ({
  primarySelectedItemKey: item ? toGalleryItemKey(item) : null,
  selectedItemKey: item ? toGalleryItemKey(item) : null,
  selectedItemKeys: item ? [toGalleryItemKey(item)] : [],
});

const createGallery = (items: GalleryItem[], selected: GalleryItem | null): GalleryStateView => ({
  anchoredWindowPage: 0,
  boards: [],
  compareImageKey: null,
  galleryView: 'images',
  isComparisonActive: false,
  items,
  page: 0,
  projectBoardId: null,
  revealTargetPage: null,
  searchTerm: '',
  selectedBoardId: 'board-a',
  ...selectionOf(selected),
  selectionStarredOnly: false,
  semanticImageQuery: null,
  semanticSearchText: null,
  // At the default density the harness shows three columns and a few rows of 400 items.
  settings: getGallerySettings({ paginationMode: 'paginated' }),
  starredOnly: false,
});

interface HarnessState {
  confirmDeletion: boolean;
  galleryMounted: boolean;
  followedSessionId: string | null;
  gallery: GalleryStateView;
  /** The widget's persisted values; the selection as the workbench keeps it. */
  galleryValues: Record<string, unknown>;
  listing: GalleryListingState;
  sessions: QueueProgressSession[];
  strip: GalleryItem[];
}

let state: HarnessState;
const listeners = new Set<() => void>();
const setState = (next: Partial<HarnessState>) => {
  state = { ...state, ...next };
  listeners.forEach((listener) => listener());
};
const setGallery = (next: Partial<GalleryStateView>) => setState({ gallery: { ...state.gallery, ...next } });
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};
const getState = () => state;

const select = (item: GalleryItem | null) =>
  setState({ followedSessionId: null, gallery: { ...state.gallery, ...selectionOf(item) } });

const actions = {
  loadMore: vi.fn(),
  selectItem: select,
  selectItemRange: (refs: GalleryItemRef[], primary: GalleryItem) =>
    setGallery({ selectedItemKey: toGalleryItemKey(primary), selectedItemKeys: refs.map(toGalleryItemKey) }),
  setCompareItem: vi.fn(),
  setStarredOnly: vi.fn(),
  // As the workbench toggles: an added item becomes the primary; removing the primary hands it on.
  toggleItemInSelection: (item: GalleryItem, nextPrimary: GalleryItem | null) => {
    const key = toGalleryItemKey(item);
    const { selectedItemKey, selectedItemKeys } = state.gallery;

    setGallery(
      selectedItemKeys.includes(key)
        ? {
            selectedItemKey:
              selectedItemKey === key ? (nextPrimary ? toGalleryItemKey(nextPrimary) : null) : selectedItemKey,
            selectedItemKeys: selectedItemKeys.filter((candidate) => candidate !== key),
          }
        : { selectedItemKey: key, selectedItemKeys: [...selectedItemKeys, key] }
    );
  },
  updateSettings: vi.fn(),
  uploadFiles: vi.fn(),
} as unknown as GalleryActions;

/**
 * Deletion as the workbench applies it: the items leave the listing and the selection at once, and the neighbour is
 * selected once the backend confirms.
 */
const applyDeletion = async (refs: GalleryItemRef[]) => {
  const removed = new Set(refs.map(toGalleryItemKey));
  const before = state.gallery.items;
  const primaryKey = state.gallery.selectedItemKey;

  setGallery({ items: before.filter((item) => !removed.has(toGalleryItemKey(item))), ...selectionOf(null) });
  await new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, 20);
  });

  const successor = primaryKey ? getGalleryDeletionSuccessor(before, primaryKey, removed) : null;

  select(before.find((item) => successor !== null && toGalleryItemKey(item) === toGalleryItemKey(successor)) ?? null);
};
const confirmationRef = createRef<RequestDeletionConfirmation>();
const openedInPreview: string[] = [];
const itemActions = {
  // As the workbench opens one: select the item, then reveal Preview elsewhere in the layout.
  openItemInPreview: (item: GalleryItem) => {
    select(item);
    openedInPreview.push(item.name);
  },
  deleteItems: (refs: GalleryItemRef[], options?: { returnFocus?: () => HTMLElement | null }) =>
    state.confirmDeletion
      ? confirmationRef.current!(refs, () => applyDeletion(refs), options?.returnFocus)
      : applyDeletion(refs),
  // The listing holds unstarred items only, so starring moves an item into the strip and back.
  setItemsStarred: (refs: GalleryItemRef[], starred: boolean) => {
    const keys = new Set(refs.map(toGalleryItemKey));
    const moved = [...state.strip, ...state.gallery.items]
      .filter((item) => keys.has(toGalleryItemKey(item)))
      .map((item) => ({ ...item, starred }));
    const rest = (items: GalleryItem[]) => items.filter((item) => !keys.has(toGalleryItemKey(item)));

    setState({
      gallery: {
        ...state.gallery,
        items: starred ? rest(state.gallery.items) : [...moved, ...rest(state.gallery.items)],
      },
      strip: starred ? [...rest(state.strip), ...moved] : rest(state.strip),
    });
    return Promise.resolve();
  },
};

/** The production confirmation dialog, which returns focus where the request says, else to whatever opened it. */
const DeletionConfirmationHost = ({ ref }: { ref: Ref<RequestDeletionConfirmation> }) => {
  const { dialog, requestDeletionConfirmation } = useDeletionConfirmation();

  useImperativeHandle(ref, () => requestDeletionConfirmation, [requestDeletionConfirmation]);

  return dialog;
};

let gallerySource: WidgetContributionSource;
const createGalleryRuntime = (extensions: ExtensionRegistry) => ({
  commands: {
    register: (command: Parameters<ExtensionRegistry['commands']['register']>[0]) =>
      extensions.commands.register({ ...command, source: gallerySource }),
  },
  hotkeys: {
    register: (hotkey: Parameters<ExtensionRegistry['hotkeys']['register']>[0]) =>
      extensions.hotkeys.register({ ...hotkey, scope: 'widget', source: gallerySource }),
  },
});

const noop = () => {};

const GalleryRegion = ({ runtime }: { runtime: ReturnType<typeof createGalleryRuntime> }) => {
  const { followedSessionId, gallery, galleryMounted, galleryValues, listing, sessions, strip } = useSyncExternalStore(
    subscribe,
    getState
  );
  // Memoized per scope, as the widget derives it.
  const filter = useMemo(() => ({ boardId: gallery.selectedBoardId }), [gallery.selectedBoardId]);
  const adapter = {
    ImageContextMenu: () => null,
    ItemActionsProvider: ({ children }: { children: ReactNode }) => children,
    antialiasProgressImages: false,
    followProgressSession: (id: string) => setState({ followedSessionId: id }),
    followedProgressSessionId: followedSessionId,
    gallery: { clearSelection: () => select(null), setPage: noop },
    galleryValues,
    getItemLabel: () => Promise.resolve(null),
    liveFollowEnabled: followedSessionId !== null,
    pinnedProgressSessionId: followedSessionId,
    progressSessions: sessions,
  } as unknown as GalleryUiAdapter;
  const contextValue = {
    actions,
    boardsState: READY_LISTING,
    filter,
    gallery,
    isWindowTruncated: false,
    itemActions,
    listing,
    loadedItems: mergeGalleryLoadedItems(strip, gallery.items),
    projectName: 'Project',
    region: 'right',
    runtime,
    starredStrip: {
      items: strip,
      state: { ...READY_LISTING, status: strip.length > 0 ? 'ready' : 'empty' },
      total: strip.length,
    },
  } as unknown as GalleryWidgetContextValue;

  return (
    <Box
      data-hotkey-widget-instance-id={gallerySource.instanceId}
      data-hotkey-widget-region="right"
      data-hotkey-widget-type-id="gallery"
      data-testid="gallery-region"
      h="full"
      minH="0"
      {...useFocusRegionProps('right')}
    >
      <GalleryUiProvider adapter={adapter}>
        <GalleryWidgetContext value={contextValue}>{galleryMounted ? <GalleryImageGrid /> : null}</GalleryWidgetContext>
      </GalleryUiProvider>
    </Box>
  );
};

/** Another docked region holding ordinary controls. */
const OtherRegion = () => (
  <Box {...useFocusRegionProps('left')}>
    <input aria-label="Prompt" />
    <button type="button">Elsewhere</button>
  </Box>
);

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (ms = 30) =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, ms);
    });
  });

/** Polls an observable condition instead of waiting a fixed time for scrolling, deletion or a dialog's exit. */
const until = (assertion: () => void) => act(() => vi.waitFor(assertion, { interval: 20, timeout: 2000 }));

const press = async (keys: string) => {
  await act(() => userEvent.keyboard(keys));
  await settle();
};

const viewport = () => host!.querySelector<HTMLElement>('[data-part="viewport"]')!;
const thumbnail = (name: string) =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="Select ${name} for preview"]`);
const button = (name: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent === name)!;
/** The thumbnail holding keyboard focus, by item name; null when focus is anywhere else. */
const focusedThumbnail = () =>
  document.activeElement instanceof HTMLButtonElement && document.activeElement.closest('[role="listitem"]') !== null
    ? document.activeElement.getAttribute('aria-label')!.replace(/^Select (.*) for preview$/, '$1')
    : null;
const selectedKey = () => state.gallery.selectedItemKey;
const selectedKeys = () => state.gallery.selectedItemKeys;
const isInView = (element: Element) => {
  const bounds = viewport().getBoundingClientRect();
  const rect = element.getBoundingClientRect();

  return rect.bottom > bounds.top && rect.top < bounds.bottom;
};

const renderGrid = async (
  items: GalleryItem[],
  {
    selected = items[0] ?? null,
    settings,
    ...rest
  }: Partial<Omit<HarnessState, 'gallery' | 'listing'>> & {
    selected?: GalleryItem | null;
    settings?: Partial<GallerySettings>;
  } = {}
) => {
  const gallery = createGallery(items, selected);

  state = {
    confirmDeletion: false,
    followedSessionId: null,
    gallery: { ...gallery, settings: { ...gallery.settings, ...settings } },
    galleryMounted: true,
    galleryValues: {},
    listing: READY_LISTING,
    sessions: [],
    strip: [],
    ...rest,
  };
  const runtime = createGalleryRuntime(runtimeMocks.extensions);

  await act(() => {
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient!}>
            <WorkbenchFocusProvider>
              <WorkbenchHotkeyRuntime />
              <DeletionConfirmationHost ref={confirmationRef} />
              <Box display="grid" gridTemplateColumns="200px 400px" gridTemplateRows="100%" h="full">
                <OtherRegion />
                <GalleryRegion runtime={runtime} />
              </Box>
              <button type="button">After the gallery</button>
            </WorkbenchFocusProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  await settle(50);
};

/** Tabs from the other region's last control into the gallery. */
const tabIntoGallery = async () => {
  button('Elsewhere').focus();
  await act(() => userEvent.tab());
  await settle();
};

beforeEach(() => {
  vi.clearAllMocks();
  openedInPreview.length = 0;
  accountLifecycle.activate('grid-focus-user');
  runtimeMocks.store = createWorkbenchStore();
  runtimeMocks.extensions = createExtensionRegistry();
  const project = runtimeMocks.store.getSnapshot().activeProject;

  gallerySource = {
    instanceId: project.widgetRegions.right.activeInstanceId,
    projectId: project.id,
    region: 'right',
    typeId: 'gallery',
  };
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  host.style.cssText = 'height:360px;left:0;position:fixed;top:0;width:600px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  host?.remove();
  queryClient?.clear();
  listeners.clear();
  host = null;
  root = null;
  queryClient = null;
});

describe('Gallery grid keyboard focus', () => {
  it('moves focus with the selection as the arrows carry it past the rendered rows', async () => {
    await renderGrid(createItems(400));
    await tabIntoGallery();
    expect(focusedThumbnail()).toBe('image-0.png');

    for (let step = 0; step < 30; step += 1) {
      await press('{ArrowDown}');
    }

    expect(selectedKey()).toBe('image:image-90.png');
    expect(focusedThumbnail()).toBe('image-90.png');
    expect(isInView(document.activeElement!)).toBe(true);
    // The rows focus started in are gone: the grid really virtualized them away.
    expect(thumbnail('image-0.png')).toBeNull();
  });

  it('activates the tile the arrows reached, not the one focus started on', async () => {
    await renderGrid(createItems(400));
    await tabIntoGallery();

    await press('{ArrowRight}');
    await press('{ArrowRight}');
    await press(' ');

    expect(focusedThumbnail()).toBe('image-2.png');
    expect(selectedKeys()).toEqual(['image:image-2.png']);
    expect(openedInPreview).toEqual([]);

    // Enter is not a gallery hotkey: it reaches the focused tile alone, which opens it.
    await press('{ArrowRight}');
    await press('{Enter}');

    expect(openedInPreview).toEqual(['image-3.png']);
    expect(selectedKeys()).toEqual(['image:image-3.png']);
  });

  it('keeps the focused tile when the wheel scrolls it away, and resumes from it on the next arrow', async () => {
    await renderGrid(createItems(400));
    await tabIntoGallery();
    const focused = document.activeElement!;

    await act(() => userEvent.wheel(viewport(), { delta: { y: 600 }, times: 8 }));
    await until(() => expect(isInView(focused)).toBe(false));

    expect(viewport().scrollTop).toBeGreaterThan(2000);
    expect(document.activeElement).toBe(focused);
    expect(selectedKey()).toBe('image:image-0.png');

    await press('{ArrowRight}');

    expect(selectedKey()).toBe('image:image-1.png');
    expect(focusedThumbnail()).toBe('image-1.png');
    expect(isInView(document.activeElement!)).toBe(true);
  });

  it('does not load more because the kept focus row sits at the end of the listing', async () => {
    const items = createItems(120);
    await renderGrid(items, { selected: items[119]!, settings: { paginationMode: 'infinite' } });

    // The selection's row stays mounted as the Tab stop, but the top of the listing is what is in view.
    expect(thumbnail('image-119.png')).not.toBeNull();
    expect(actions.loadMore).not.toHaveBeenCalled();

    await act(() => {
      viewport().scrollTop = viewport().scrollHeight;
    });
    await until(() => expect(actions.loadMore).toHaveBeenCalled());
  });

  it.each([
    ['a middle tile', 'image-1.png', 'image-2.png'],
    ['the last tile', 'image-5.png', 'image-4.png'],
  ])('hands focus from %s to the neighbour selected after deleting it', async (_, deleted, neighbour) => {
    const items = createItems(6);
    await renderGrid(items, { selected: items.find((item) => item.name === deleted)! });
    await tabIntoGallery();
    expect(focusedThumbnail()).toBe(deleted);

    await press('{Delete}');
    await until(() => expect(selectedKey()).toBe(`image:${neighbour}`));

    expect(thumbnail(deleted)).toBeNull();
    expect(focusedThumbnail()).toBe(neighbour);

    await press('{ArrowLeft}');

    expect(focusedThumbnail()).not.toBe(neighbour);
    expect(selectedKey()).toBe(`image:${focusedThumbnail()}`);
  });

  it('hands focus to the neighbour when deletion is confirmed in a dialog', async () => {
    const items = createItems(6);
    await renderGrid(items, { confirmDeletion: true, selected: items[1]! });
    await tabIntoGallery();

    await press('{Delete}');
    await until(() => expect(document.querySelector('[role="alertdialog"]')).not.toBeNull());
    button('Delete').focus();
    await press('{Enter}');
    await until(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());

    expect(selectedKey()).toBe('image:image-2.png');
    expect(focusedThumbnail()).toBe('image-2.png');
  });

  it('leaves focus alone when a tile it is not in leaves the grid', async () => {
    const items = createItems(6);
    await renderGrid(items, { selected: items[1]! });
    (document.activeElement as HTMLElement | null)?.blur();
    expect(document.activeElement).toBe(document.body);

    // Nothing holds focus, and the deleted tile never did: the gallery must not take it.
    await act(() => runtimeMocks.extensions.commands.executeForSource('gallery.deleteSelection', gallerySource));
    await until(() => expect(selectedKey()).toBe('image:image-2.png'));

    expect(thumbnail('image-1.png')).toBeNull();
    expect(document.activeElement).toBe(document.body);
  });

  it('lets focus go when the whole grid unmounts', async () => {
    await renderGrid(createItems(6));
    await tabIntoGallery();
    expect(focusedThumbnail()).toBe('image-0.png');

    await act(() => setState({ galleryMounted: false }));
    await settle();

    expect(host!.querySelector('[data-part="viewport"]')).toBeNull();
    expect(document.activeElement).toBe(document.body);
  });

  it('keeps a starred tile focused as the star moves it into the strip', async () => {
    await renderGrid(createItems(6), { strip: [createItem('starred-0.png', { starred: true })] });
    const star = host!.querySelector<HTMLButtonElement>('button[aria-label="Star image-2.png"]')!;

    // A pointer press focuses the star, which is click-focusable though not a Tab stop.
    await act(() => {
      star.focus();
      star.click();
    });
    await settle();

    expect(focusedThumbnail()).toBe('image-2.png');
    expect(document.activeElement?.closest('[data-gallery-section="starred"]')).not.toBeNull();
    expect(selectedKey()).toBe('image:image-0.png');
  });

  it('steps from a starred selection beyond the strip, as Preview does, not from the first starred tile', async () => {
    const strip = [0, 1, 2].map((index) => createItem(`starred-${index}.png`, { starred: true }));
    const beyond = createItem('starred-40.png', { starred: true });
    const items = createItems(6);
    // The strip's bound left the selection out, and the listing holds unstarred items only: no tile shows it.
    await renderGrid(items, {
      galleryValues: { selectedBoardId: 'board-a', selectedImage: beyond },
      selected: null,
      strip,
    });
    expect(thumbnail('starred-40.png')).toBeNull();
    button('Elsewhere').focus();

    await act(() => runtimeMocks.extensions.commands.executeForSource('gallery.galleryNavRight', gallerySource));
    await settle();

    expect(selectedKey()).toBe('image:image-0.png');
  });

  it("starts from the new board's first starred tile when the starred selection stayed on the previous board", async () => {
    const starredA = createItem('starred-a.png', { starred: true });
    await renderGrid(createItems(6), {
      // Selecting stamps the listing the selection was made in.
      galleryValues: {
        selectedBoardId: 'board-a',
        selectedImage: starredA,
        selectedImageName: 'image:starred-a.png',
        selectedImageQuery: {
          boardId: 'board-a',
          galleryView: 'images',
          imageOrderDir: 'DESC',
          page: 0,
          paginationMode: 'paginated',
          searchTerm: '',
          starredOnly: false,
        },
      },
      selected: starredA,
      strip: [starredA],
    });

    // As the board switch leaves it: the selection set clears, but the primary selection and its stamp persist.
    const stripB = [0, 1, 2].map((index) =>
      createItem(`starred-b-${index}.png`, { boardId: 'board-b', starred: true })
    );
    await act(() =>
      setState({
        gallery: { ...state.gallery, items: createItems(6, 'other'), selectedBoardId: 'board-b', ...selectionOf(null) },
        galleryValues: { ...state.galleryValues, selectedBoardId: 'board-b', selectedImageNames: [] },
        strip: stripB,
      })
    );
    await settle();
    button('Elsewhere').focus();

    await act(() => runtimeMocks.extensions.commands.executeForSource('gallery.galleryNavRight', gallerySource));
    await settle();

    expect(selectedKey()).toBe('image:starred-b-0.png');
  });

  it('holds focus on the named grid while a new board loads, then resumes on that board', async () => {
    await renderGrid(createItems(40), { selected: createItem('image-7.png') });
    await tabIntoGallery();
    expect(focusedThumbnail()).toBe('image-7.png');

    await act(() =>
      setState({
        gallery: { ...state.gallery, items: [], selectedBoardId: 'board-b' },
        listing: { ...READY_LISTING, status: 'loading' },
      })
    );
    await settle();

    expect(document.activeElement).toBe(viewport());
    expect(viewport().getAttribute('role')).toBe('group');
    expect(viewport().getAttribute('aria-label')).toBe('Gallery content');
    expect(viewport().matches(':focus-visible')).toBe(true);
    expect(getComputedStyle(viewport()).outlineStyle).not.toBe('none');

    await act(() =>
      setState({ gallery: { ...state.gallery, items: createItems(40, 'other') }, listing: READY_LISTING })
    );
    await settle();

    // With the old selection off this board, Tab and the arrows both start from its first tile.
    expect(host!.querySelector('[role="listitem"] button[tabindex="0"]')?.getAttribute('aria-label')).toBe(
      'Select other-0.png for preview'
    );
    await press('{ArrowRight}');

    expect(selectedKey()).toBe('image:other-0.png');
    expect(focusedThumbnail()).toBe('other-0.png');
  });

  it('leaves focus where it is when the selection moves from outside the grid', async () => {
    await renderGrid(createItems(40));
    const elsewhere = button('Elsewhere');
    elsewhere.focus();

    // An arrow aimed at another region is not the gallery's.
    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-0.png');

    // The command palette runs the gallery's command with focus elsewhere.
    await act(() => runtimeMocks.extensions.commands.executeForSource('gallery.galleryNavRight', gallerySource));
    await settle();

    expect(selectedKey()).toBe('image:image-1.png');
    expect(document.activeElement).toBe(elsewhere);
  });

  it('gives the grid one thumbnail Tab stop, the selection, even when it is scrolled out of view', async () => {
    const items = createItems(400);
    await renderGrid(items, { selected: items[200]! });
    expect(viewport().scrollTop).toBe(0);

    await tabIntoGallery();

    expect(focusedThumbnail()).toBe('image-200.png');
    expect(isInView(document.activeElement!)).toBe(true);

    // Neither the other tiles nor their star toggles are Tab stops.
    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(button('After the gallery'));

    await act(() => userEvent.tab({ shift: true }));
    expect(focusedThumbnail()).toBe('image-200.png');
  });

  it('extends the selection from its anchor with Shift and the arrows', async () => {
    const items = createItems(40);
    await renderGrid(items, { selected: items[1]! });
    await tabIntoGallery();

    await press('{Shift>}{ArrowRight}{/Shift}');
    await press('{Shift>}{ArrowRight}{/Shift}');
    await until(() => expect(selectedKey()).toBe('image:image-3.png'));

    expect(selectedKeys()).toEqual(['image:image-1.png', 'image:image-2.png', 'image:image-3.png']);
    expect(focusedThumbnail()).toBe('image-3.png');

    // Down a row of three keeps the anchor; back up shrinks the range to it again.
    await press('{Shift>}{ArrowDown}{/Shift}');
    await until(() => expect(selectedKeys()).toHaveLength(6));
    expect(selectedKeys().at(-1)).toBe('image:image-6.png');

    await press('{Shift>}{ArrowUp}{/Shift}');
    await until(() => expect(selectedKeys()).toHaveLength(3));
    expect(focusedThumbnail()).toBe('image-3.png');

    // A plain arrow replaces the range.
    await press('{ArrowLeft}');

    expect(selectedKeys()).toEqual(['image:image-2.png']);
    expect(focusedThumbnail()).toBe('image-2.png');
  });

  it('treats a Shift range across the starred strip as Shift+click does', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    const items = createItems(40);
    await renderGrid(items, { selected: strip[1]!, strip });
    await tabIntoGallery();
    await act(() => userEvent.tab());
    expect(focusedThumbnail()).toBe('starred-1.png');

    await press('{Shift>}{ArrowDown}{/Shift}');
    await until(() => expect(selectedKey()).toBe('image:image-1.png'));
    const byKeyboard = selectedKeys();

    expect(focusedThumbnail()).toBe('image-1.png');

    await act(() => select(strip[1]!));
    await act(() =>
      thumbnail('image-1.png')!.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }))
    );
    await until(() => expect(selectedKey()).toBe('image:image-1.png'));

    expect(selectedKeys()).toEqual(byKeyboard);
  });

  it('moves focus alone with the mod key, then toggles the focused tile with a modified Space', async () => {
    const items = createItems(40);
    await renderGrid(items);
    await tabIntoGallery();

    await press('{Control>}{ArrowRight}{ArrowRight}{/Control}');

    expect(focusedThumbnail()).toBe('image-2.png');
    expect(selectedKeys()).toEqual(['image:image-0.png']);

    // Up and down as well: the app binds Ctrl+Up/Down to prompt weight, but inside the gallery its keys win.
    await press('{Control>}{ArrowDown}{/Control}');
    expect(focusedThumbnail()).toBe('image-5.png');
    await press('{Control>}{ArrowUp}{/Control}');
    expect(focusedThumbnail()).toBe('image-2.png');
    expect(selectedKeys()).toEqual(['image:image-0.png']);

    // No hotkey is bound by default: the browser activates the focused button as a Ctrl+click, which toggles it.
    await press('{Control>} {/Control}');

    expect(selectedKeys()).toEqual(['image:image-0.png', 'image:image-2.png']);
    expect(focusedThumbnail()).toBe('image-2.png');

    await press('{Control>}{ArrowRight} {/Control}');
    await press('{Control>} {/Control}');

    expect(selectedKeys()).toEqual(['image:image-0.png', 'image:image-2.png']);

    // The focused tile, not the selection, is where Tab comes back to.
    await act(() => userEvent.tab());
    await act(() => userEvent.tab({ shift: true }));
    expect(focusedThumbnail()).toBe('image-3.png');

    // A plain arrow steps from focus and replaces the selection.
    await press('{ArrowRight}');

    expect(selectedKeys()).toEqual(['image:image-4.png']);
    expect(focusedThumbnail()).toBe('image-4.png');
  });

  it('toggles the focused tile with a key the user assigned to the toggle command', async () => {
    const items = createItems(40);
    await renderGrid(items);
    await tabIntoGallery();

    // Unbound by default, so the key does nothing until it is assigned as Settings assigns it.
    await press('{Control>}{ArrowRight}{/Control}');
    await press('x');
    expect(selectedKeys()).toEqual(['image:image-0.png']);

    await act(() => patchWorkbenchPreferences({ customHotkeys: { 'gallery.toggleFocusedInSelection': ['x'] } }));
    await press('x');

    expect(selectedKeys()).toEqual(['image:image-0.png', 'image:image-1.png']);
    expect(focusedThumbnail()).toBe('image-1.png');

    await press('x');

    expect(selectedKeys()).toEqual(['image:image-0.png']);
  });

  it('follows the arrows between the starred strip, running sessions and the listing', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    await renderGrid(createItems(40), {
      selected: strip[0]!,
      sessions: [createSession('queued', 'queued'), createSession('running', 'running')],
      strip,
    });
    // The strip's disclosure comes first, then the one thumbnail Tab stop.
    await tabIntoGallery();
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Collapse starred items');
    await act(() => userEvent.tab());
    expect(focusedThumbnail()).toBe('starred-0.png');

    // Down from the strip skips the queued session, which cannot be followed.
    await press('{ArrowDown}');

    expect(state.followedSessionId).toBe('running');
    expect(document.activeElement?.getAttribute('data-gallery-session-id')).toBe('running');

    await press('{ArrowDown}');

    expect(selectedKey()).toBe('image:image-1.png');
    expect(focusedThumbnail()).toBe('image-1.png');

    await press('{ArrowUp}');
    await press('{ArrowUp}');

    expect(selectedKey()).toBe('image:starred-1.png');
    expect(focusedThumbnail()).toBe('starred-1.png');
  });

  it('steps Shift and mod arrows from a followed in-progress tile to its neighbour, never the first starred tile', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    await renderGrid(createItems(40), { sessions: [createSession('running', 'running')], strip });
    await act(() => userEvent.click(thumbnail('image-1.png')!));
    await settle();

    // Up from the listing's first row follows the running tile above it.
    await press('{ArrowUp}');
    expect(state.followedSessionId).toBe('running');
    expect(document.activeElement?.getAttribute('data-gallery-session-id')).toBe('running');

    // Focus alone moves past the tile to the next thumbnail, which follows it in the listing, then on along it.
    await press('{Control>}{ArrowRight}{/Control}');
    expect(focusedThumbnail()).toBe('image-0.png');
    await press('{Control>}{ArrowRight}{/Control}');
    expect(focusedThumbnail()).toBe('image-1.png');
    await press('{Control>}{ArrowRight}{/Control}');
    expect(focusedThumbnail()).toBe('image-2.png');
    expect(selectedKey()).toBe('image:image-1.png');

    await press('{Control>}{ArrowLeft}{ArrowLeft}{/Control}');
    expect(focusedThumbnail()).toBe('image-0.png');

    await press('{ArrowUp}');
    expect(state.followedSessionId).toBe('running');

    // A range runs from the selection to that same neighbour.
    await press('{Shift>}{ArrowRight}{/Shift}');
    await until(() => expect(selectedKey()).toBe('image:image-0.png'));
    expect(selectedKeys()).toEqual(['image:image-0.png', 'image:image-1.png']);
    expect(focusedThumbnail()).toBe('image-0.png');
  });

  it('steps from a clicked thumbnail while a run is followed, also after a resubmit follows it again', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    const items = createItems(40);
    // The user's sequence: a run is followed live when they click a thumbnail, which pauses following.
    await renderGrid(items, {
      followedSessionId: 'running',
      selected: null,
      sessions: [createSession('running', 'running')],
      strip,
    });
    await act(() => userEvent.click(thumbnail('image-4.png')!));
    await settle();
    expect(state.followedSessionId).toBeNull();

    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-5.png');

    // Invoking again follows the new run while the thumbnail keeps keyboard focus: the arrows step from that tile,
    // whether the In progress section is shown or collapsed.
    for (const [collapsed, expected] of [
      [false, 'image-6.png'],
      [true, 'image-7.png'],
    ] as const) {
      await act(() => {
        setGallery({ settings: { ...state.gallery.settings, progressSectionCollapsed: collapsed } });
        setState({ followedSessionId: 'running' });
      });
      await press('{ArrowRight}');
      expect(selectedKey()).toBe(`image:${expected}`);
      expect(focusedThumbnail()).toBe(expected);
    }
  });

  it('steps from the selection while the followed tile is collapsed out of the grid', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    const items = createItems(40);
    await renderGrid(items, {
      followedSessionId: 'running',
      selected: items[4]!,
      sessions: [createSession('running', 'running')],
      settings: { progressSectionCollapsed: true },
      strip,
    });
    // Collapsing the section leaves focus on its disclosure, outside every thumbnail.
    await act(() => host!.querySelector<HTMLButtonElement>('[data-progress-disclosure]')!.focus());

    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-5.png');
    expect(focusedThumbnail()).toBe('image-5.png');

    // Each resubmit follows live again; with focus off the thumbnails and the tile still hidden, the arrows keep
    // stepping from the selection.
    for (const expected of ['image-6.png', 'image-7.png']) {
      await act(() => {
        host!.querySelector<HTMLButtonElement>('[data-progress-disclosure]')!.focus();
        setState({ followedSessionId: 'running' });
      });
      await press('{ArrowRight}');
      expect(selectedKey()).toBe(`image:${expected}`);
    }

    // Shown, the followed tile is where the arrows start, as in Preview: right of it is the listing's first tile.
    await act(() => setGallery({ settings: { ...state.gallery.settings, progressSectionCollapsed: false } }));
    await act(() => {
      host!.querySelector<HTMLButtonElement>('[data-progress-disclosure]')!.focus();
      setState({ followedSessionId: 'running' });
    });
    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-0.png');
  });

  it('starts from the first thumbnail in view, not the first starred one, when no cursor is on screen', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    await renderGrid(createItems(400), { selected: null, strip });
    // Scrolled well past the strip with nothing selected and focus on the grid itself, off every thumbnail.
    await act(() => {
      viewport().scrollTop = 3000;
    });
    await until(() => expect(thumbnail('image-0.png')).toBeNull());
    await act(() => viewport().focus());

    await press('{ArrowRight}');

    const landed = selectedKey()!.replace(/^image:/, '');
    expect(landed).not.toBe('starred-0.png');
    expect(isInView(thumbnail(landed)!)).toBe(true);
    expect(focusedThumbnail()).toBe(landed);
  });

  it('returns focus to the thumbnail Tab stop, not the first starred tile, when the last followed run ends', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    await renderGrid(createItems(40), { sessions: [createSession('running', 'running')], strip });
    await act(() => userEvent.click(thumbnail('image-1.png')!));
    await settle();
    // With no selection, the Tab stop is the thumbnail focus was last on.
    await press('{Escape}');
    expect(selectedKey()).toBeNull();
    expect(focusedThumbnail()).toBe('image-1.png');

    await press('{ArrowUp}');
    expect(document.activeElement?.getAttribute('data-gallery-session-id')).toBe('running');

    // The run finishes; its tile, holding focus, leaves with the whole In progress section.
    await act(() => setState({ followedSessionId: null, sessions: [] }));
    await until(() => expect(focusedThumbnail()).toBe('image-1.png'));
  });

  it('keeps stepping from the selection as results arrive above it and as it moves into the strip', async () => {
    const strip = [createItem('starred-0.png', { starred: true }), createItem('starred-1.png', { starred: true })];
    const items = createItems(40);
    await renderGrid(items, { sessions: [createSession('running', 'running')], strip });
    await act(() => userEvent.click(thumbnail('image-4.png')!));
    await settle();

    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-5.png');

    // The run finishes: its result lands at the top of the listing and its tile leaves, shifting every row.
    const result = createItem('result-0.png');
    await act(() => setState({ gallery: { ...state.gallery, items: [result, ...state.gallery.items] }, sessions: [] }));
    await until(() => expect(focusedThumbnail()).toBe('image-5.png'));

    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:image-6.png');
    expect(focusedThumbnail()).toBe('image-6.png');

    // Starred, the selection moves to the strip's end; the next step is from there, onto the listing's first tile.
    await press('.');
    await until(() => expect(document.activeElement?.closest('[data-gallery-section="starred"]')).not.toBeNull());
    await press('{ArrowRight}');
    expect(selectedKey()).toBe('image:result-0.png');
    expect(focusedThumbnail()).toBe('result-0.png');
  });
});
