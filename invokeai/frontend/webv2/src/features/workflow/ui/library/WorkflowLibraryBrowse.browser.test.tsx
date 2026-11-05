import type { ListWorkflowsParams, WorkflowLibraryListItem, WorkflowLibraryPage } from '@features/workflow/data/api';
import type { WorkflowUiAdapter, WorkflowPreferences } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GalleryHostProvider } from '@features/gallery/picker';
import * as browse from '@features/workflow/data/libraryBrowseStore';
import { invalidateWorkflowLibraryCache } from '@features/workflow/data/libraryCache';
import { WorkflowUiProvider, WorkflowGraphPreviewProvider } from '@features/workflow/ui/WorkflowUiContext';
import { setWorkflowLibraryTab } from '@features/workflow/ui/workflowUiStore';
import { createProjectGraph } from '@features/workflow/utility';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

const transport = vi.hoisted(() => ({
  list: vi.fn<(params: ListWorkflowsParams) => Promise<WorkflowLibraryPage>>(),
  record: vi.fn(),
}));
vi.mock('@features/workflow/data/api', async (original) => ({
  ...(await original<Record<string, unknown>>()),
  listLibraryWorkflows: (params: ListWorkflowsParams) => transport.list(params),
  getLibraryWorkflowRecord: (id: string) => transport.record(id),
  getWorkflowTagCounts: () => Promise.resolve({ fixture1000: 1000 }),
  getAllWorkflowTags: () => Promise.resolve(['fixture1000']),
}));
const templatesSnapshot = {
  status: 'loaded',
  error: null,
  templates: {
    integer: {
      category: 'test',
      classification: 'stable',
      description: '',
      inputs: {},
      outputs: {},
      nodePack: 'invokeai',
      outputType: 'integer_output',
      tags: [],
      title: 'integer',
      type: 'integer',
      useCache: true,
      version: '1.0.0',
    },
  },
};
vi.mock('@features/workflow/data/templates', async (original) => ({
  ...(await original<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => templatesSnapshot,
  useInvocationTemplatesSnapshot: () => templatesSnapshot,
}));
vi.mock('@features/identity', async (original) => ({
  ...(await original<Record<string, unknown>>()),
  useAuthSession: () => ({ multiuserEnabled: false, phase: 'ready', user: null }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { count?: number; source?: string }) =>
      key === 'workflowLibrary.nodeCount'
        ? `${values?.count} nodes`
        : key === 'graphPreview.compiledFrom'
          ? `Compiled from ${values?.source}`
          : key,
  }),
}));

const graph = createProjectGraph('project-workflow');
const project = {
  id: 'project',
  activeWorkflow: { document: graph },
  activeWorkflowId: graph.id,
  projectGraph: graph,
  workflows: [{ document: graph }],
  galleryValues: {},
  workflowValues: {},
  isWorkflowRunning: false,
};
const port = <T,>(snapshot: T) => ({ getSnapshot: () => snapshot, subscribe: () => () => {} });
const adapter = {
  project: port(project),
  preferences: port({
    themeId: 'classic',
    workflowLibrarySort: 'updated',
    workflowLibraryThumbnailFit: 'cover',
  } as WorkflowPreferences),
  commands: {},
  getProjectGraph: () => graph,
  setLibraryPreferences: vi.fn(),
  notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  openAddModels: vi.fn(),
} as unknown as WorkflowUiAdapter;
const galleryHost = { galleryValues: {}, notifications: { add: vi.fn(), reportError: vi.fn() }, projectName: '' };
const previewAdapter = {
  openDocumentInNewProject: vi.fn(),
  getRoute: () => null,
  invoke: () => Promise.resolve(false),
  focusSource: vi.fn(),
  openWorkflowEditor: vi.fn(),
};
let root: Root;
let host: HTMLDivElement;
let count = 1000;
const item = (index: number): WorkflowLibraryListItem => ({
  workflow_id: `wf-${index}`,
  name: `Workflow ${index}`,
  revision: 1,
  category: 'user',
  description: '',
  tags: '',
});
const list = ({ page, perPage = 20 }: ListWorkflowsParams) =>
  Promise.resolve({
    page,
    pages: Math.ceil(count / perPage),
    total: count,
    items: Array.from({ length: Math.min(perPage, Math.max(0, count - page * perPage)) }, (_, i) =>
      item(page * perPage + i)
    ),
  });
const card = (index: number) => document.querySelector<HTMLElement>(`[data-workflow-card="wf-${index}"]`);
const viewport = () =>
  card(0)?.closest<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]') ??
  document.querySelector<HTMLElement>('[data-part="viewport"][aria-label="workflowLibrary.title"]');
const waitFor = (assert: () => void) =>
  act(async () => {
    await vi.waitFor(assert, { timeout: 5000 });
  });
const open = async () => {
  setWorkflowLibraryTab('user');
  await act(async () => {
    await Promise.resolve();
    root.render(
      <StrictMode>
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <GalleryHostProvider host={galleryHost}>
              <WorkflowGraphPreviewProvider adapter={previewAdapter}>
                <WorkflowLibraryDialog isOpen onOpenChange={vi.fn()} />
              </WorkflowGraphPreviewProvider>
            </GalleryHostProvider>
          </WorkflowUiProvider>
        </ChakraProvider>
      </StrictMode>
    );
  });
  await waitFor(() => expect(card(0)).not.toBeNull());
};
beforeEach(async () => {
  accountLifecycle.invalidate();
  count = 1000;
  transport.list.mockReset().mockImplementation(list);
  transport.record.mockReset().mockImplementation((id: string) =>
    Promise.resolve({
      ...item(Number(id.slice(3))),
      created_at: '',
      updated_at: '',
      user_id: '',
      is_public: false,
      workflow: { id, name: id, nodes: [], edges: [], meta: { version: '3.0.0', category: 'user' } },
    })
  );
  await page.viewport(1280, 800);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

it('shows middle-range failure and successful Retry inside the viewport with bounded cards', async () => {
  await open();
  transport.list.mockImplementation((params: ListWorkflowsParams) => {
    if (params.page >= 20) {
      return Promise.reject(new Error('middle offline'));
    }
    return list(params);
  });
  const scroll = viewport()!;
  await act(() => {
    scroll.scrollTop = scroll.scrollHeight / 2;
    scroll.dispatchEvent(new Event('scroll'));
  });
  await waitFor(() => expect(document.querySelector('[role="alert"]')?.textContent).toContain('middle offline'));
  const retry = document.querySelector<HTMLElement>('[role="alert"] button')!;
  const rect = retry.getBoundingClientRect();
  const box = scroll.getBoundingClientRect();
  expect(rect.top).toBeGreaterThanOrEqual(box.top);
  expect(rect.bottom).toBeLessThanOrEqual(box.bottom);
  transport.list.mockImplementation(list);
  await act(() => userEvent.click(retry));
  await waitFor(() => expect(browse.getWorkflowLibraryBrowseSnapshot().error).toBeNull());
  expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBeGreaterThan(300);
  expect(document.querySelectorAll('[data-workflow-card]').length).toBeLessThan(48);
});
it('Retry rereads a failed resident refresh while keeping existing cards', async () => {
  count = 5;
  await open();
  transport.list.mockRejectedValueOnce(new Error('refresh offline'));
  await act(() => browse.refreshWorkflowLibraryBrowse());
  expect(card(0)).not.toBeNull();
  const calls = transport.list.mock.calls.length;
  const retry = document.querySelector<HTMLElement>('[role="alert"] button')!;
  await act(() => userEvent.click(retry));
  await waitFor(() => expect(browse.getWorkflowLibraryBrowseSnapshot().error).toBeNull());
  expect(transport.list.mock.calls.length).toBeGreaterThan(calls);
  expect(browse.getWorkflowLibraryBrowseSnapshot().total).toBe(5);
});
it.each(['coalesced', 'superseded', 'canceled intent', 'wheel intent'])(
  'finishes delayed real dialog boundary navigation: %s',
  async (scenario) => {
    await open();
    await act(() => browse.loadWorkflowLibraryRange(0, 59));
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    transport.list.mockImplementation(async (params: ListWorkflowsParams) => {
      await gate;
      return list(params);
    });
    // Scroll to the last resident row, then focus its last menu control.
    const scroll = viewport()!;
    await act(() => {
      scroll.scrollTop = 19 * 200;
      scroll.dispatchEvent(new Event('scroll'));
    });
    await waitFor(() => expect(document.querySelector('[data-workflow-card-menu="wf-59"]')).not.toBeNull());
    const source = document.querySelector<HTMLElement>('[data-workflow-card-menu="wf-59"]')!;
    await act(() => source.focus());
    const demand = browse.loadWorkflowLibraryRange(60, 65);
    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(source);
    const superseding = scenario === 'superseded' ? browse.loadWorkflowLibraryRange(40, 65) : null;
    if (scenario === 'canceled intent') {
      await act(() => userEvent.keyboard('{Shift>}{Tab}{/Shift}'));
      expect(document.activeElement).toBe(card(59));
    }
    if (scenario === 'wheel intent') {
      await act(() => {
        scroll.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -400 }));
        scroll.scrollTop -= 400;
        scroll.dispatchEvent(new Event('scroll'));
      });
      await act(() => {
        scroll.scrollTop += 400;
        scroll.dispatchEvent(new Event('scroll'));
      });
    }
    await act(async () => {
      release();
      await demand;
      await superseding;
    });
    await waitFor(() => expect(card(60)).not.toBeNull());
    await waitFor(() => {
      if (scenario === 'wheel intent') {
        expect(document.activeElement).toBe(source);
      } else {
        expect(document.activeElement).toBe(card(scenario === 'canceled intent' ? 59 : 60));
      }
    });
  }
);

const detail = () => document.querySelector<HTMLElement>('[data-workflow-detail="wf-0"]')!;
const previewButton = () =>
  Array.from(detail().querySelectorAll('button')).find(
    (button) => button.textContent === 'workflowLibrary.previewGraph'
  )!;

it('uses the newer selected metadata and document when the resident list arrives later', async () => {
  count = 1;
  await open();
  await waitFor(() => expect(previewButton().disabled).toBe(false));
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  transport.list.mockImplementation(async (params: ListWorkflowsParams) => {
    await gate;
    const result = await list(params);
    return { ...result, items: result.items.map((entry) => ({ ...entry, name: 'Older resident refresh' })) };
  });
  transport.record.mockResolvedValue({
    ...item(0),
    name: 'Fresh selection',
    description: 'Fresh description',
    revision: 2,
    workflow: {
      name: 'Fresh document',
      description: 'Fresh description',
      edges: [],
      nodes: [0, 1].map((index) => ({
        id: `fresh-node-${index}`,
        type: 'invocation',
        position: { x: index * 200, y: 0 },
        data: { type: 'integer', inputs: {} },
      })),
    },
  });
  await act(() => invalidateWorkflowLibraryCache('wf-0'));
  await waitFor(() => expect(browse.getWorkflowLibraryBrowseSnapshot().selectedEntry?.item.revision).toBe(2));
  await act(() => release());
  await waitFor(() => {
    expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded');
    expect(card(0)?.textContent).toContain('Older resident refresh');
    expect(detail().textContent).toContain('Fresh selection');
    expect(detail().textContent).toContain('Fresh description');
    expect(previewButton().disabled).toBe(false);
  });
  await act(() => userEvent.click(previewButton()));
  await waitFor(() => expect(document.body.textContent).toContain('Compiled from Fresh selection'));
  await waitFor(() => expect(document.querySelectorAll('.react-flow__node')).toHaveLength(2));
  await act(() => userEvent.click(page.getByRole('tab', { name: 'common.json', exact: true })));
  await waitFor(() => {
    expect(document.body.textContent).toContain('fresh-node-0');
    expect(document.body.textContent).toContain('fresh-node-1');
  });
});

it('shows the selected transient error over equal-revision resident pending state and recovers', async () => {
  count = 1;
  await open();
  await waitFor(() => expect(previewButton().disabled).toBe(false));
  transport.list.mockImplementation(async (params: ListWorkflowsParams) => {
    const result = await list(params);
    return { ...result, items: result.items.map((entry) => ({ ...entry, name: 'Refreshed resident' })) };
  });
  transport.record.mockRejectedValue(new Error('selected record offline'));
  await act(() => invalidateWorkflowLibraryCache('wf-0'));
  await waitFor(() => {
    expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded');
    expect(card(0)?.textContent).toContain('Refreshed resident');
    expect(detail().textContent).toContain('selected record offline');
    expect(detail().textContent).toContain('Workflow 0');
    expect(previewButton().disabled).toBe(true);
  });
  transport.record.mockResolvedValue({
    ...item(0),
    workflow: {
      name: 'Recovered document',
      nodes: [],
      edges: [],
    },
  });
  await act(() => browse.refreshWorkflowLibraryBrowse());
  await waitFor(() => {
    expect(detail().textContent).not.toContain('selected record offline');
    expect(detail().textContent).toContain('Workflow 0');
    expect(previewButton().disabled).toBe(false);
  });
});

const firstVisibleRow = (scroll: HTMLElement) => {
  const top = scroll.getBoundingClientRect().top;
  const rows = Array.from(scroll.querySelectorAll<HTMLElement>('[data-index]'));
  const row = rows.find((element) => element.getBoundingClientRect().bottom > top)!;
  const rect = row.getBoundingClientRect();
  return {
    id: row.querySelector<HTMLElement>('[data-workflow-card]')?.dataset.workflowCard,
    offset: top - rect.top,
    fraction: (top - rect.top) / rect.height,
    columns: row.querySelectorAll('[data-workflow-card]').length,
  };
};
const settle = () =>
  act(
    () =>
      new Promise((resolve) => {
        setTimeout(resolve, 250);
      })
  );
const scrollTo = async (scroll: HTMLElement, top: number) => {
  await act(() => {
    scroll.scrollTop = top;
    scroll.dispatchEvent(new Event('scroll'));
  });
  await act(
    () =>
      new Promise((resolve) => {
        setTimeout(resolve, 120);
      })
  );
};
it.each([
  [1280, 800, 3],
  [512, 800, 2],
  [320, 400, 1],
])(
  'returns to the measured deep row at %ix%i (%i columns) through the real browse owner',
  async (width, height, columns) => {
    await open();
    const scroll = viewport()!;
    for (let top = 0; top <= 39600; top += 1200) {
      await scrollTo(scroll, top);
      expect(document.querySelectorAll('[data-workflow-card]').length).toBeLessThan(48);
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries.length).toBeLessThanOrEqual(60);
    }
    await settle();
    const before = firstVisibleRow(scroll);
    expect(Number(before.id?.slice(3))).toBeGreaterThan(400);
    expect(before.offset).toBeGreaterThan(10);
    const requests = transport.list.mock.calls.length;
    await act(() => userEvent.click(page.getByRole('button', { name: 'fixture1000', exact: false })));
    await settle();
    await page.viewport(width!, height!);
    await settle();
    const all = page.getByRole('button', { name: 'workflowLibrary.allTag', exact: false });
    const modal = document.querySelector<HTMLElement>('[role=dialog]')!;
    const outerOffset = modal.scrollTop;
    await act(() => userEvent.click(all));
    await settle();
    const restored = viewport()!;
    const expectedIndex = Math.floor(Number(before.id?.slice(3)) / columns!) * columns!;
    await waitFor(() => expect(firstVisibleRow(restored).id).toBe(`wf-${expectedIndex}`));
    expect(firstVisibleRow(restored).columns).toBe(columns);
    expect(Math.abs(firstVisibleRow(restored).fraction - before.fraction)).toBeLessThan(0.01);
    if (width === 1280) {
      expect(Math.abs(firstVisibleRow(restored).offset - before.offset)).toBeLessThan(3);
    }
    expect(document.activeElement).toBe(all.element());
    expect(modal.scrollTop).toBe(outerOffset);
    expect(document.querySelectorAll('[data-workflow-card]').length).toBeLessThan(48);
    expect(browse.getWorkflowLibraryBrowseRetention().metadataEntries).toBeLessThanOrEqual(540);
    // Only the new tag view needs a listing; returning must not replace its cached deep window with page zero.
    expect(transport.list.mock.calls.length - requests).toBe(1);
    await scrollTo(restored, 0);
    await waitFor(() => expect(card(0)).not.toBeNull());
    expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBe(0);
  },
  30000
);

it.each(['finish', 'retry', 'wheel', 'pointer', 'key', 'filter', 'rapid return', 'account', 'unmount'])(
  'handles a delayed missing-window return: %s',
  async (scenario) => {
    await open();
    const scroll = viewport()!;
    await scrollTo(scroll, 39600);
    await settle();
    const before = firstVisibleRow(scroll);
    const anchorIndex = Number(before.id?.slice(3));
    // A real position read may finish just as a user leaves. Retained metadata and the last visible anchor
    // are independent; returning must wait for the anchor's window instead of restoring placeholder geometry.
    await act(async () => {
      await browse.loadWorkflowLibraryRange(700, 705);
      browse.setWorkflowLibraryBrowseFilter({ tag: 'fixture1000' });
    });
    await waitFor(() => expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded'));
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let failReturn = scenario === 'retry';
    transport.list.mockImplementation(async (params: ListWorkflowsParams) => {
      if (!params.tags?.length && params.page > 0) {
        await gate;
        if (failReturn) {
          throw new Error('return window offline');
        }
      }
      return list(params);
    });
    transport.list.mockClear();
    await act(() => browse.setWorkflowLibraryBrowseFilter({ tag: null }));
    await waitFor(() => expect(transport.list.mock.calls.length).toBe(3));
    expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBe(700);
    const returned = viewport()!;
    expect(returned.scrollTop).toBe(0);
    expect(firstVisibleRow(returned).id).toBeUndefined();
    const focus = document.activeElement;
    let canceledOffset = 0;
    if (['wheel', 'pointer', 'key'].includes(scenario)) {
      await act(() => {
        returned.dispatchEvent(
          scenario === 'wheel'
            ? new WheelEvent('wheel', { bubbles: true, deltaY: -1200 })
            : scenario === 'pointer'
              ? new PointerEvent('pointerdown', { bubbles: true })
              : new KeyboardEvent('keydown', { bubbles: true, key: 'PageUp' })
        );
      });
      await scrollTo(returned, 0);
      canceledOffset = returned.scrollTop;
    } else if (scenario === 'filter' || scenario === 'rapid return') {
      await act(() => browse.setWorkflowLibraryBrowseFilter({ tag: 'fixture1000' }));
      if (scenario === 'rapid return') {
        await act(() => browse.setWorkflowLibraryBrowseFilter({ tag: null }));
        await waitFor(() => expect(transport.list.mock.calls.length).toBe(6));
      }
    } else if (scenario === 'account') {
      await act(() => accountLifecycle.invalidate());
    } else if (scenario === 'unmount') {
      await act(() => root.render(null));
    }
    await act(async () => {
      release();
      await gate;
    });
    await settle();
    if (scenario === 'retry') {
      await waitFor(() =>
        expect(document.querySelector('[role=alert]')?.textContent).toContain('return window offline')
      );
      expect(transport.list.mock.calls).toHaveLength(3);
      failReturn = false;
      await act(() => userEvent.click(document.querySelector<HTMLElement>('[role=alert] button')!));
      await settle();
    }
    if (scenario === 'finish' || scenario === 'rapid return') {
      const active = viewport()!;
      await waitFor(() => expect(firstVisibleRow(active).id).toBe(before.id));
      expect(Math.abs(firstVisibleRow(active).offset - before.offset)).toBeLessThan(3);
      expect(document.activeElement).toBe(focus);
      const pages = transport.list.mock.calls.map(([params]) => params.page);
      expect(scenario === 'rapid return' ? pages.slice(3) : pages).toEqual([
        Math.floor((anchorIndex - 6) / 20),
        Math.floor((anchorIndex - 6) / 20) + 1,
        Math.floor((anchorIndex - 6) / 20) + 2,
      ]);
    } else if (scenario === 'retry') {
      expect(browse.getWorkflowLibraryBrowseSnapshot().error).toBeNull();
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries.length).toBeGreaterThan(0);
      expect(returned.scrollTop).toBe(0);
      expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBe(0);
      expect(firstVisibleRow(returned).id).toBe('wf-0');
    } else if (['wheel', 'pointer', 'key'].includes(scenario)) {
      expect(returned.scrollTop).toBe(canceledOffset);
      expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBe(0);
      expect(firstVisibleRow(returned).id).toBe('wf-0');
    } else if (scenario === 'filter') {
      expect(browse.getWorkflowLibraryBrowseSnapshot().filter.tag).toBe('fixture1000');
      expect(firstVisibleRow(viewport()!).id).toBe('wf-0');
      expect(browse.getWorkflowLibraryBrowseSnapshot().startIndex).toBe(0);
    } else if (scenario === 'account') {
      expect(browse.getWorkflowLibraryBrowseSnapshot().scrollAnchor).toBeUndefined();
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries).toHaveLength(0);
    } else {
      expect(returned.isConnected).toBe(false);
    }
    const canceled = !['finish', 'unmount'].includes(scenario);
    expect(transport.list.mock.calls.slice(0, 3).every(([params]) => params.signal?.aborted)).toBe(canceled);
    expect(transport.list.mock.calls).toHaveLength(
      ['wheel', 'pointer', 'key', 'rapid return', 'retry'].includes(scenario) ? 6 : 3
    );
    if (scenario === 'account' || scenario === 'unmount') {
      expect(browse.getWorkflowLibraryBrowseRetention().parsedDocuments).toBe(0);
    }
    expect(browse.getWorkflowLibraryBrowseRetention().metadataEntries).toBeLessThanOrEqual(540);
    expect(document.querySelectorAll('[data-workflow-card]').length).toBeLessThan(48);
  },
  15000
);
