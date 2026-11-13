/* eslint-disable react-perf/jsx-no-new-object-as-prop */
import type * as BrowseStoreModule from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, type ComponentProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { WorkflowLibraryList } from './WorkflowLibraryList';

const store = vi.hoisted(() => ({
  loadWorkflowLibraryRange: vi.fn((_start: number, _end: number, _force?: boolean) => Promise.resolve(true)),
  refreshWorkflowLibraryBrowse: vi.fn(() => Promise.resolve()),
  rememberWorkflowLibraryScroll: vi.fn(),
  setWorkflowLibraryVisibleDemand: vi.fn(),
}));

vi.mock('@features/workflow/data/libraryBrowseStore', async (original) => ({
  ...(await original<typeof BrowseStoreModule>()),
  loadWorkflowLibraryRange: store.loadWorkflowLibraryRange,
  refreshWorkflowLibraryBrowse: store.refreshWorkflowLibraryBrowse,
  rememberWorkflowLibraryScroll: store.rememberWorkflowLibraryScroll,
  setWorkflowLibraryVisibleDemand: store.setWorkflowLibraryVisibleDemand,
}));
vi.mock('@features/workflow/react', async (original) => ({
  ...(await original<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => ({ error: null, status: 'loaded', templates: {} }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const graph = createProjectGraph('list-fixture');
const ADAPTER = {
  project: {
    getSnapshot: () => ({ activeWorkflowId: graph.id, id: 'project', workflows: [{ document: graph }] }),
    subscribe: () => () => {},
  },
} as unknown as WorkflowUiAdapter;
const FILTER = { direction: 'desc', place: 'user', search: '', sort: 'updated', tags: [] } as const;

const entry = (index: number): WorkflowLibraryEntry => ({
  enrichment: { document: graph, nodeCount: 3, requirements: { primaryBase: null, requirements: [] }, status: 'ready' },
  item: { category: 'user', description: '', name: `Workflow ${index}`, revision: 1, workflow_id: `wf-${index}` },
  tags: [],
});
const page = (start: number, count = 20) => Array.from({ length: count }, (_, offset) => entry(start + offset));

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;height:400px;width:720px;';
  document.body.append(host);
  root = createRoot(host);
  store.loadWorkflowLibraryRange.mockClear();
  store.refreshWorkflowLibraryBrowse.mockClear();
  store.rememberWorkflowLibraryScroll.mockClear();
  store.setWorkflowLibraryVisibleDemand.mockClear();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

type ListProps = ComponentProps<typeof WorkflowLibraryList>;

const render = async (props: Partial<ListProps>) => {
  const merged: ListProps = {
    entries: page(0),
    error: null,
    onContextMenu: vi.fn(),
    onOpen: vi.fn(),
    onPreview: vi.fn(),
    onSelect: vi.fn(),
    selectedWorkflowId: null,
    status: 'loaded',
    window: { filter: FILTER, startIndex: 0, total: 100 },
    ...props,
  };
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <WorkflowUiProvider adapter={ADAPTER}>
          <WorkflowLibraryList {...merged} />
        </WorkflowUiProvider>
      </ChakraProvider>
    )
  );
};

const primaryOf = (workflowId: string) => document.getElementById(`workflow-library-card-${workflowId}`);
const rowAt = (index: number) => host.querySelector<HTMLElement>(`[data-list-row="position:${index}"]`);

describe('WorkflowLibraryList', () => {
  it('keeps the selected row with its name while its page is evicted, until the workflow is removed', async () => {
    await render({ selectedWorkflowId: 'wf-1' });
    expect(primaryOf('wf-1')?.getAttribute('aria-current')).toBe('true');

    // The window moved on: position 1 is no longer resident, but the selection keeps its row.
    await render({
      entries: page(60),
      selectedWorkflowId: 'wf-1',
      window: { filter: FILTER, startIndex: 60, total: 100 },
    });
    expect(primaryOf('wf-1')?.textContent).toContain('Workflow 1');
    expect(rowAt(1)?.querySelector('[data-list-primary]')).toBe(primaryOf('wf-1'));
    expect(primaryOf('wf-1')?.getAttribute('aria-current')).toBe('true');

    // Removed (deleted from the rail): the kept copy goes, and the position is a placeholder again.
    await render({
      entries: page(60),
      removedWorkflowId: 'wf-1',
      selectedWorkflowId: null,
      window: { filter: FILTER, startIndex: 60, total: 100 },
    });
    expect(primaryOf('wf-1')).toBeNull();
    expect(rowAt(1)?.querySelector('[data-workflow-list-placeholder]')).not.toBeNull();
  });

  it('names placeholder rows as loading and keeps one tab stop when the selection is not resident', async () => {
    await render({
      entries: page(60),
      selectedWorkflowId: 'wf-1',
      window: { filter: FILTER, startIndex: 60, total: 100 },
    });

    const placeholder = rowAt(0)!;
    expect(placeholder.querySelector('[data-workflow-list-placeholder]')).not.toBeNull();
    expect(placeholder.querySelector('[data-list-primary]')?.getAttribute('aria-label')).toBe(
      'workflowLibrary.loading'
    );
    // The selection was made outside the window: the List's tab stop falls back to its first row, a real button.
    const stops = host.querySelectorAll('[data-workflow-library-list] [tabindex="0"]');
    expect(stops).toHaveLength(1);
    expect(stops[0]).toBe(placeholder.querySelector('[data-list-primary]'));
  });

  it('keeps the browser’s own menu off placeholder rows without raising the actions menu', async () => {
    const onContextMenu = vi.fn();
    await render({ entries: page(60), onContextMenu, window: { filter: FILTER, startIndex: 60, total: 100 } });
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 20 });

    await act(() => rowAt(0)!.querySelector('[data-list-primary]')!.dispatchEvent(event));
    expect(event.defaultPrevented).toBe(true);
    expect(onContextMenu).not.toHaveBeenCalled();
  });

  it('keeps every row’s width when the pointer reveals Open and Preview', async () => {
    await render({ selectedWorkflowId: 'wf-0' });
    const button = primaryOf('wf-1')!;
    const width = button.getBoundingClientRect().width;
    const open = rowAt(1)!.querySelector<HTMLElement>('[data-row-action="open"]')!;
    expect(getComputedStyle(open).visibility).toBe('hidden');

    await userEvent.hover(button);
    expect(getComputedStyle(open).visibility).toBe('visible');
    expect(button.getBoundingClientRect().width).toBe(width);
    // Every row's controls take the same room, so the date column stays aligned.
    expect(primaryOf('wf-2')!.getBoundingClientRect().width).toBe(width);

    await userEvent.unhover(button);
    expect(button.getBoundingClientRect().width).toBe(width);
  });

  it('names the last run once in the This-project view: as the date column when sorted by it, else in the facts', async () => {
    const ran = { ...entry(0), item: { ...entry(0).item, last_run_at: '2026-01-01T00:00:00Z' } };
    await render({ entries: [ran], trailingField: 'last_run_at', window: undefined });
    expect(primaryOf('wf-0')?.textContent).toContain('workflowLibrary.nodeCount');
    expect(primaryOf('wf-0')?.textContent).not.toContain('workflowLibrary.ranWhen');

    await render({ entries: [ran], trailingField: 'updated_at', window: undefined });
    expect(primaryOf('wf-0')?.textContent).toContain('workflowLibrary.ranWhen');
  });

  it('releases closed-list demand without overwriting its scroll anchor, and reacquires on reopen', async () => {
    await render({});
    await expect.poll(() => store.setWorkflowLibraryVisibleDemand.mock.calls.at(-1)?.[0].length).toBeGreaterThan(0);
    const viewport = host.querySelector<HTMLElement>('[data-list-viewport]')!;
    await act(() => {
      viewport.scrollTop = 270;
      viewport.dispatchEvent(new Event('scroll'));
    });
    await expect.poll(() => store.rememberWorkflowLibraryScroll.mock.calls.length).toBeGreaterThan(0);
    const anchor = store.rememberWorkflowLibraryScroll.mock.calls.at(-1);

    // Closing retains the collection while the dialog's hidden layout collapses its viewport.
    await render({ isActive: false });
    host.style.display = 'none';
    await expect.poll(() => store.setWorkflowLibraryVisibleDemand.mock.calls.at(-1)?.[0]).toEqual([]);
    store.loadWorkflowLibraryRange.mockClear();
    await act(() => {
      viewport.scrollTop = 0;
      viewport.dispatchEvent(new Event('scroll'));
    });
    expect(store.rememberWorkflowLibraryScroll.mock.calls.at(-1)).toEqual(anchor);
    expect(store.loadWorkflowLibraryRange).not.toHaveBeenCalled();

    host.style.display = 'flex';
    await render({ isActive: true });
    await expect.poll(() => store.setWorkflowLibraryVisibleDemand.mock.calls.at(-1)?.[0].length).toBeGreaterThan(0);
  });

  it('persists the final fractional scroll offset after its viewport detaches', async () => {
    await render({});
    const viewport = host.querySelector<HTMLElement>('[data-list-viewport]')!;
    await act(() => {
      viewport.scrollTop = 281;
      viewport.dispatchEvent(new Event('scroll'));
    });
    store.rememberWorkflowLibraryScroll.mockClear();
    await act(() => root.render(null));
    expect(store.rememberWorkflowLibraryScroll).toHaveBeenLastCalledWith(FILTER, {
      fraction: 11 / 54,
      index: 5,
    });
  });

  it('retries a failed range in place, forcing a reread of the positions on screen', async () => {
    await render({ error: 'middle offline' });
    const retry = host.querySelector<HTMLElement>('[role="alert"] button')!;
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('middle offline');

    await act(() => userEvent.click(retry));
    expect(store.loadWorkflowLibraryRange).toHaveBeenLastCalledWith(0, expect.any(Number), true);
  });

  it('reloads a failed first listing, busy until it settles', async () => {
    let settle!: () => void;
    store.refreshWorkflowLibraryBrowse.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          settle = resolve;
        })
    );
    await render({
      entries: [],
      error: 'offline',
      status: 'error',
      window: { filter: FILTER, startIndex: 0, total: 0 },
    });
    const retry = () => host.querySelector<HTMLButtonElement>('[role="alert"] button')!;
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('offline');

    await act(() => userEvent.click(retry()));
    expect(store.refreshWorkflowLibraryBrowse).toHaveBeenCalledTimes(1);
    expect(store.loadWorkflowLibraryRange).not.toHaveBeenCalledWith(expect.anything(), expect.anything(), true);
    expect(retry().disabled).toBe(true);

    await act(async () => {
      settle();
      await Promise.resolve();
    });
    expect(retry().disabled).toBe(false);
  });
});
