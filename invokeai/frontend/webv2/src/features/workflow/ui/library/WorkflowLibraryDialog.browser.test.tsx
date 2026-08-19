import type {
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibraryEntryEnrichment,
} from '@features/workflow/data/libraryBrowseStore';

import { ChakraProvider } from '@chakra-ui/react';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

// The browse store is Task 5's; this suite owns the *dialog*, so the store is
// replaced by a real external store the test drives directly plus spies for
// its four commands. Using a real store (not a stub hook) keeps the dialog's
// subscription, selector equality, and re-render path under test.
const browse = vi.hoisted(() => ({
  ensureWorkflowLibraryBrowseLoaded: vi.fn(() => Promise.resolve()),
  // Assigned by the module factory below, which owns the store instance.
  getSnapshot: null as null | (() => WorkflowLibraryBrowseSnapshot),
  loadNextWorkflowLibraryPage: vi.fn(),
  setSnapshot: null as null | ((snapshot: WorkflowLibraryBrowseSnapshot) => void),
  setWorkflowLibraryBrowseFilter: vi.fn(),
}));

vi.mock('@features/workflow/data/libraryBrowseStore', async () => {
  const { createExternalStore } = await import('@platform/state/externalStore');
  const store = createExternalStore<WorkflowLibraryBrowseSnapshot>({
    entries: [],
    error: null,
    filter: { category: 'user', search: '', tag: null },
    page: 0,
    pages: 0,
    status: 'idle',
    tagCounts: [],
    total: 0,
    userTotal: null,
  });

  browse.getSnapshot = store.getSnapshot;
  browse.setSnapshot = store.setSnapshot;

  return {
    ensureWorkflowLibraryBrowseLoaded: browse.ensureWorkflowLibraryBrowseLoaded,
    getWorkflowLibraryBrowseSnapshot: store.getSnapshot,
    loadNextWorkflowLibraryPage: browse.loadNextWorkflowLibraryPage,
    setWorkflowLibraryBrowseFilter: browse.setWorkflowLibraryBrowseFilter,
    useWorkflowLibraryBrowseSelector: store.useSelector,
  };
});

// The load sequence (fetch → parse → replace → toasts) has its own behavior
// contract; this suite only asserts the dialog *invokes* it and reflects its
// phase in the busy overlay.
const loader = vi.hoisted(() => ({
  load: vi.fn(() => Promise.resolve()),
  phase: { current: 'idle' as 'applying' | 'fetching' | 'idle' },
}));

vi.mock('./useLoadLibraryWorkflow', () => ({
  useLoadLibraryWorkflow: () => ({ load: loader.load, loadPhase: loader.phase.current }),
}));

// The real i18n client fetches en.json over HTTP, which this browser test
// never boots. Stub `t` with the English strings this dialog renders (plus
// the `_one`/`_other` plural forms) so assertions check real copy.
const TRANSLATIONS: Record<string, string> = {
  'common.close': 'Close',
  'workflowLibrary.allTag': 'All',
  'workflowLibrary.applying': 'Applying workflow…',
  'workflowLibrary.browse': 'Browse',
  'workflowLibrary.empty': 'No workflows match these filters.',
  'workflowLibrary.fetching': 'Fetching workflow…',
  'workflowLibrary.loading': 'Loading workflows…',
  'workflowLibrary.loadingMore': 'Loading more…',
  'workflowLibrary.nodeCount_one': '{{count}} node',
  'workflowLibrary.nodeCount_other': '{{count}} nodes',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.searchPlaceholder': 'Search names, tags, or descriptions',
  'workflowLibrary.title': 'Workflows',
  'workflowLibrary.yours': 'Yours',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replaceAll(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

const translate = (key: string, options?: Record<string, unknown>): string => {
  const count = options?.count;
  const plural = typeof count === 'number' ? TRANSLATIONS[`${key}_${count === 1 ? 'one' : 'other'}`] : undefined;

  return interpolate(plural ?? TRANSLATIONS[key] ?? key, options);
};

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: translate }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const EMPTY_DOCUMENT = createProjectGraph('library-fixture');

const readyEnrichment = (nodeCount: number, primaryBase: string | null): WorkflowLibraryEntryEnrichment => ({
  document: EMPTY_DOCUMENT,
  nodeCount,
  requirements: { primaryBase, requirements: [] },
  status: 'ready',
});

const entry = (
  workflowId: string,
  name: string,
  enrichment: WorkflowLibraryEntryEnrichment,
  extras: { tags?: readonly string[]; thumbnailUrl?: string } = {}
): WorkflowLibraryEntry => ({
  enrichment,
  item: {
    category: 'user',
    description: `${name} description`,
    name,
    thumbnail_url: extras.thumbnailUrl ?? null,
    workflow_id: workflowId,
  },
  tags: extras.tags ?? [],
});

const PORTRAIT = entry('wf-portrait', 'Portrait Studio', readyEnrichment(12, 'sdxl'), {
  tags: ['portrait'],
  thumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
});
const LANDSCAPE = entry('wf-landscape', 'Landscape Pass', readyEnrichment(7, 'flux'), { tags: ['landscape'] });
const SKETCH = entry('wf-sketch', 'Sketch To Render', { status: 'pending' });
const UPSCALE = entry(
  'wf-upscale',
  'Upscale Chain',
  { message: 'unreadable', status: 'error' },
  {
    tags: ['upscale'],
  }
);

const LOADED_SNAPSHOT: WorkflowLibraryBrowseSnapshot = {
  entries: [PORTRAIT, LANDSCAPE, SKETCH, UPSCALE],
  error: null,
  filter: { category: 'user', search: '', tag: null },
  page: 0,
  pages: 2,
  status: 'loaded',
  tagCounts: [
    { count: 2, tag: 'portrait' },
    { count: 1, tag: 'upscale' },
  ],
  total: 24,
  userTotal: 4,
};

const withSnapshot = (patch: Partial<WorkflowLibraryBrowseSnapshot>): WorkflowLibraryBrowseSnapshot => ({
  ...LOADED_SNAPSHOT,
  ...patch,
});

describe('WorkflowLibraryDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  /**
   * Chakra's `SegmentGroup` tracks its indicator rect from an observer that
   * commits React state a task after the commit that armed it. Under
   * `StrictMode`'s mount/unmount/remount that lands in the gap *between* two
   * `act` calls, where React's act queue is null — which is what raised the
   * "update ... was not wrapped in act(...)" warnings. Awaiting this **inside**
   * the same `act` scope as the render keeps the queue open across that gap.
   */
  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const renderDialog = async (isOpen = true) => {
    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowLibraryDialog isOpen={isOpen} onOpenChange={onOpenChange} />
          </ChakraProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const openWith = async (snapshot: WorkflowLibraryBrowseSnapshot) => {
    await act(() => browse.setSnapshot?.(snapshot));
    await renderDialog();
  };

  /** Drains the microtask + timer queue the store probe and debounce settle on. */
  const wait = (ms: number) =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      });
    });

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);

  const clickText = async (text: string) => {
    const button = buttonWithText(text);
    expect(button).not.toBeUndefined();

    await act(() => button?.click());
  };

  const clickSegment = async (value: string) => {
    const label = document.querySelector<HTMLInputElement>(`input[value="${value}"]`)?.closest('label');
    expect(label).not.toBeNull();

    await act(async () => {
      label?.click();
      await settleFrame();
    });
  };

  const gridViewport = () => document.querySelector<HTMLElement>('[role="region"][aria-label="Workflows"]');

  /**
   * Real layout in a `Dialog` portal gives no deterministic scroll geometry,
   * so the viewport's metrics are defined directly — the assertion is about
   * the dialog's near-bottom threshold, not the browser's layout engine.
   */
  const scrollTo = async (scrollTop: number) => {
    const viewport = gridViewport();
    expect(viewport).not.toBeNull();

    for (const [property, value] of [
      ['clientHeight', 400],
      ['scrollHeight', 4000],
      ['scrollTop', scrollTop],
    ] as const) {
      Object.defineProperty(viewport, property, { configurable: true, value, writable: true });
    }

    await act(async () => {
      viewport?.dispatchEvent(new Event('scroll'));
      // The ScrollArea machine reacts to the scroll a task later; see
      // `settleFrame` for why that has to stay inside this `act` scope.
      await settleFrame();
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();
    browse.loadNextWorkflowLibraryPage.mockClear();
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    loader.load.mockClear();
    loader.phase.current = 'idle';
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('renders one card per entry with its name and enriched node count', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(cards()).toHaveLength(4);

    const text = document.body.textContent ?? '';

    expect(text).toContain('Portrait Studio');
    expect(text).toContain('12 nodes');
    expect(text).toContain('SDXL');
    expect(text).toContain('Landscape Pass');
    expect(text).toContain('7 nodes');
    expect(text).toContain('FLUX');
  });

  it('shows a quiet placeholder while enrichment is pending and omits the badges when it failed', async () => {
    await openWith(LOADED_SNAPSHOT);

    const pending = card('wf-sketch');
    expect(pending?.textContent).toContain('Sketch To Render');
    expect(pending?.textContent).not.toContain('nodes');
    expect(pending?.querySelector('[data-enrichment-placeholder]')).not.toBeNull();

    // A failed enrichment is a quiet omission, never an error-styled card.
    const failed = card('wf-upscale');
    expect(failed?.textContent).toContain('Upscale Chain');
    expect(failed?.textContent).not.toContain('nodes');
    expect(failed?.querySelector('[data-enrichment-placeholder]')).toBeNull();
  });

  it('renders the thumbnail when present and the not-run-yet placeholder when absent', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.querySelector('img')).not.toBeNull();
    expect(card('wf-landscape')?.querySelector('img')).toBeNull();
    expect(card('wf-landscape')?.textContent).toContain('Not run yet');
  });

  it('kicks the first load when it opens and leaves the store alone while closed', async () => {
    await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
    await renderDialog(false);

    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();

    await renderDialog(true);

    // Times are not asserted: StrictMode double-invokes mount effects, and the
    // store's own single-flight is what makes that a no-op.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalled();
  });

  it('switches to the bundled defaults once when the account has no workflows of its own', async () => {
    await openWith(withSnapshot({ userTotal: 0 }));
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ category: 'default', tag: null });
  });

  it('leaves the category alone when the account already has workflows', async () => {
    await openWith(LOADED_SNAPSHOT);
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();
  });

  it('resets the tag when the category segment changes', async () => {
    await openWith(withSnapshot({ filter: { category: 'user', search: '', tag: 'portrait' } }));

    await clickSegment('default');

    // The store applies filter patches literally, so clearing the tag on a
    // category switch is the dialog's job.
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ category: 'default', tag: null });
  });

  it('selects and clears a tag from the chip row', async () => {
    await openWith(LOADED_SNAPSHOT);

    await clickText('portrait2');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ tag: 'portrait' });

    await clickText('All');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ tag: null });
  });

  it('pushes the search term to the store once, after the debounce', async () => {
    await openWith(LOADED_SNAPSHOT);

    const input = document.querySelector<HTMLInputElement>('input[type="search"]');
    expect(input).not.toBeNull();

    // React tracks the input's value through its own setter, so assigning
    // `.value` directly would be swallowed as a no-op change.
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    for (const value of ['p', 'ph', 'pho', 'phot', 'photo']) {
      await act(async () => {
        setValue?.call(input, value);
        input?.dispatchEvent(new Event('input', { bubbles: true }));
        await Promise.resolve();
      });
    }

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();

    await wait(400);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledTimes(1);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ search: 'photo' });
  });

  it('requests the next page only once the grid is scrolled near the bottom', async () => {
    await openWith(LOADED_SNAPSHOT);

    await scrollTo(100);

    expect(browse.loadNextWorkflowLibraryPage).not.toHaveBeenCalled();

    await scrollTo(3200);

    expect(browse.loadNextWorkflowLibraryPage).toHaveBeenCalled();
  });

  it('auto-selects the first entry and moves the selection when it disappears', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.getAttribute('aria-pressed')).toBe('true');
    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('false');

    await act(() => card('wf-landscape')?.click());

    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('true');

    // A filter change drops the selected row; selection falls back to the head
    // of the new list rather than pointing at nothing.
    await act(() => browse.setSnapshot?.(withSnapshot({ entries: [SKETCH, UPSCALE] })));

    expect(card('wf-sketch')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('opens a workflow on double click', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => {
      card('wf-landscape')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });

    expect(loader.load).toHaveBeenCalledTimes(1);
    expect(loader.load).toHaveBeenCalledWith(LANDSCAPE.item);
  });

  it('dedupes entries that share a workflow id', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, LANDSCAPE, { ...PORTRAIT }] }));

    expect(cards()).toHaveLength(2);
    expect(document.querySelectorAll('[data-workflow-card="wf-portrait"]')).toHaveLength(1);
  });

  it('keeps the loaded grid and reports a failed load-more inline', async () => {
    await openWith(withSnapshot({ error: 'Failed to load more workflows.', status: 'error' }));

    expect(cards()).toHaveLength(4);
    expect(document.body.textContent ?? '').toContain('Failed to load more workflows.');
    expect(document.body.textContent ?? '').not.toContain('No workflows match these filters.');
  });

  it('shows the loading copy on the very first paint, before the store leaves idle', async () => {
    // The pristine store: `ensureWorkflowLibraryBrowseLoaded` only flips the
    // status to 'loading' from the open-time effect, so the first paint of a
    // freshly-booted session renders against 'idle'. That must never read as
    // "nothing matched".
    await openWith({
      entries: [],
      error: null,
      filter: { category: 'user', search: '', tag: null },
      page: 0,
      pages: 0,
      status: 'idle',
      tagCounts: [],
      total: 0,
      userTotal: null,
    });

    const text = document.body.textContent ?? '';

    expect(text).toContain('Loading workflows…');
    expect(text).not.toContain('No workflows match these filters.');
  });

  it('shows the empty state only when nothing loaded', async () => {
    await openWith(withSnapshot({ entries: [], status: 'loaded', total: 0 }));

    expect(cards()).toHaveLength(0);
    expect(document.body.textContent ?? '').toContain('No workflows match these filters.');
  });

  it('shows the busy overlay while a workflow is being applied', async () => {
    loader.phase.current = 'applying';

    await openWith(LOADED_SNAPSHOT);

    const status = document.querySelector('[role="status"]');
    expect(status?.textContent).toContain('Applying workflow…');
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
  });
});
