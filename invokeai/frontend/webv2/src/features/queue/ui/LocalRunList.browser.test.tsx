/* eslint-disable react-perf/jsx-no-new-object-as-prop */
import type { QueueReadModel } from '@features/queue/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
  queueCommands: {},
}));

import { createTestLocalRun, createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider, type QueueLocalRun, type QueueUiAdapter } from './QueueUiContext';
import { QueueWidgetView } from './QueueWidgetView';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The server holds one queued item and four finished ones; nothing it holds is listed, so only local rows show.
const serverModel: QueueReadModel = {
  current: null,
  items: [],
  next: null,
  scope: {},
  status: {
    processor: { isProcessing: false, isStarted: true },
    queue: {
      canceled: 1,
      completed: 2,
      failed: 1,
      inProgress: 0,
      pending: 1,
      queueId: 'default',
      total: 5,
      waiting: 0,
    },
  },
};

let host: HTMLDivElement;
let root: Root;
let client: QueryClient;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'width:360px;height:600px;';
  document.body.append(host);
  root = createRoot(host);
  client = new QueryClient();
  client.setQueryData(['queue'], serverModel);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

/** `listed` defaults to every run in scope: the test runs are old, so none waits out the reveal delay. */
const render = async (overrides: Partial<QueueUiAdapter>, listed?: number) => {
  const adapter = createTestQueueUiAdapter({ activeProjectId: 'project-1', ...overrides });

  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <QueueUiProvider adapter={adapter}>
              <QueueWidgetView region="right" />
            </QueueUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  // The rows load lazily with the first run; wait for the ones the widget lists.
  await expect
    .poll(() => rows().length)
    .toBe(
      listed ??
        adapter.localRuns.filter((run) => adapter.queueJobsScope === 'all' || run.projectId === adapter.activeProjectId)
          .length
    );
};

const section = () => host.querySelector<HTMLElement>('[role="list"][aria-label="Submitting"]');
const rows = () => [...(section()?.querySelectorAll<HTMLElement>(':scope > [role="listitem"]') ?? [])];
const rowFor = (title: string) => rows().find((row) => row.textContent?.includes(title))!;
const cancelFor = (title: string) =>
  rowFor(title).querySelector<HTMLButtonElement>(`button[aria-label="Cancel ${title}"]`)!;
const tab = (label: string) =>
  [...host.querySelectorAll<HTMLElement>('[role="tab"]')].find((element) => element.textContent?.startsWith(label))!;

describe('runs this tab is still submitting', () => {
  it('lists each run with why it still waits and which attempt comes next', async () => {
    await render({
      localRuns: [
        createTestLocalRun({ id: 'a', label: 'Fresh run' }),
        createTestLocalRun({
          id: 'b',
          label: 'Broken graph',
          retry: { failureCount: 7, message: 'Internal Server Error', reason: 'server-error' },
        }),
        createTestLocalRun({
          id: 'c',
          label: 'Busy server',
          retry: { failureCount: 1, message: 'Service Unavailable', reason: 'unavailable' },
        }),
        createTestLocalRun({
          id: 'd',
          label: 'Lost network',
          retry: { failureCount: 1234, message: 'Failed to fetch', reason: 'unreachable' },
        }),
        createTestLocalRun({
          id: 'e',
          label: 'Unreadable answer',
          retry: { failureCount: 2, message: 'Unexpected end of JSON input', reason: 'error' },
        }),
      ],
    });

    expect(rowFor('Fresh run').textContent).toContain('Sending to the server…');
    expect(rowFor('Broken graph').textContent).toContain('Server error, retrying (attempt 8)');
    expect(rowFor('Broken graph').textContent).toContain('Internal Server Error');
    expect(rowFor('Busy server').textContent).toContain('Server busy or unavailable, retrying (attempt 2)');
    expect(rowFor('Lost network').textContent).toContain("Can't reach the server, retrying (attempt 1,235)");
    expect(rowFor('Unreadable answer').textContent).toContain("Couldn't submit, retrying (attempt 3)");
    expect(rowFor('Unreadable answer').textContent).not.toContain("Can't reach the server");
  });

  it('announces why a run retries when that starts or its reason changes, not on every further attempt', async () => {
    const retrying = (failureCount: number, reason: NonNullable<QueueLocalRun['retry']>['reason']) =>
      createTestLocalRun({ label: 'Broken graph', retry: { failureCount, message: 'Bad answer', reason } });
    await render({ localRuns: [createTestLocalRun({ label: 'Broken graph' })] });
    // The region exists before there is news: one inserted along with its text is not announced.
    const region = host.querySelector<HTMLElement>('[role="status"][aria-live="polite"]')!;
    const lines = () => [...region.children].map((line) => line.textContent);
    let changeCount = 0;
    const changes = new MutationObserver((records) => {
      changeCount += records.length;
    });
    changes.observe(region, { characterData: true, childList: true, subtree: true });
    const takeChanges = () => {
      const count = changeCount + changes.takeRecords().length;
      changeCount = 0;
      return count;
    };

    expect(lines()).toEqual([]);

    await render({ localRuns: [retrying(1, 'unreachable')] });
    expect(lines()).toEqual(["Broken graph: Can't reach the server, retrying"]);
    expect(takeChanges()).toBeGreaterThan(0);

    // Another attempt failing the same way changes the row's attempt number, but leaves the region as it was.
    await render({ localRuns: [retrying(2, 'unreachable')] });
    expect(rowFor('Broken graph').textContent).toContain('(attempt 3)');
    expect(takeChanges()).toBe(0);

    await render({ localRuns: [retrying(3, 'server-error')] });
    expect(lines()).toEqual(['Broken graph: Server error, retrying']);
    expect(takeChanges()).toBeGreaterThan(0);

    // Switching to a filter that hides the section is no news either, and the region keeps announcing under it.
    await act(() => tab('Done').click());
    expect(takeChanges()).toBe(0);

    await render({ localRuns: [retrying(4, 'unavailable')] }, 0);
    expect(region.isConnected).toBe(true);
    expect(lines()).toEqual(['Broken graph: Server busy or unavailable, retrying']);
    expect(takeChanges()).toBeGreaterThan(0);

    await act(() => tab('All').click());
    expect(takeChanges()).toBe(0);
    changes.disconnect();
  });

  it('waits for the connection while offline, whatever the last failure was', async () => {
    await render({
      isConnected: false,
      localRuns: [
        createTestLocalRun({
          label: 'Queued offline',
          retry: { failureCount: 2, message: 'Failed to fetch', reason: 'unreachable' },
        }),
      ],
    });

    expect(rowFor('Queued offline').textContent).toContain('Offline, waiting for the connection…');
    expect(rowFor('Queued offline').textContent).not.toContain("Can't reach the server");
  });

  it("shows a long failure's first line, and the whole of it in a bounded area once the row is expanded", async () => {
    const message = `Internal Server Error\n${'Traceback line\n'.repeat(60)}`;
    await render({
      localRuns: [
        createTestLocalRun({ label: 'Broken graph', retry: { failureCount: 1, message, reason: 'server-error' } }),
      ],
    });
    const primary = rowFor('Broken graph').querySelector<HTMLButtonElement>('[data-list-primary]')!;

    expect(primary.getAttribute('aria-expanded')).toBe('false');
    expect(rowFor('Broken graph').textContent).not.toContain('Traceback');

    await act(() => primary.click());

    const detail = rowFor('Broken graph').querySelector<HTMLElement>('[role="region"]')!;
    expect(primary.getAttribute('aria-expanded')).toBe('true');
    expect(detail.getAttribute('aria-label')).toBe('Why Broken graph is retrying');
    expect(detail.textContent).toBe(message.trim());
    // Sixty lines would be far taller; the detail scrolls instead of pushing the queue down.
    expect(detail.getBoundingClientRect().height).toBeLessThanOrEqual(128);
    expect(detail.scrollHeight).toBeGreaterThan(detail.clientHeight);
  });

  it('holds back a run whose first enqueue may still land, and shows it once that outlasts a second', async () => {
    await render({ localRuns: [createTestLocalRun({ label: 'Just queued', submittedAt: Date.now() - 200 })] }, 0);

    expect(host.textContent).not.toContain('Submitting');
    expect(tab('All').textContent).toBe('All5');

    await expect.poll(() => rows().length, { timeout: 3000 }).toBe(1);
    expect(tab('All').textContent).toBe('All6');
  });

  it("keeps the reveal timer within setTimeout's range when the clock moved back since the run was queued", async () => {
    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');
    try {
      await render(
        { localRuns: [createTestLocalRun({ label: 'From the future', submittedAt: Date.now() + 2 ** 32 })] },
        0
      );
      await new Promise((resolve) => {
        setTimeout(resolve, 100);
      });

      // An out-of-range delay would fire at once and re-arm in a tight loop.
      const delays = setTimeoutSpy.mock.calls.map(([, delay]) => delay ?? 0);
      expect(delays.every((delay) => delay <= 2 ** 31 - 1)).toBe(true);
      expect(delays.filter((delay) => delay === 1000)).toHaveLength(1);
    } finally {
      setTimeoutSpy.mockRestore();
    }
  });

  it('shows a run at once after a failure or while offline, however recently it was queued', async () => {
    const now = Date.now();
    await render(
      {
        localRuns: [
          createTestLocalRun({
            id: 'a',
            label: 'Failed once',
            retry: { failureCount: 1, message: 'Bad Gateway', reason: 'unavailable' },
            submittedAt: now,
          }),
          createTestLocalRun({ id: 'b', label: 'Just queued', submittedAt: now }),
        ],
      },
      1
    );
    expect(rowFor('Failed once')).toBeDefined();

    await render({ isConnected: false, localRuns: [createTestLocalRun({ label: 'Just queued', submittedAt: now })] });
    expect(rowFor('Just queued').textContent).toContain('Offline, waiting for the connection…');
  });

  it('cancels the run it belongs to without asking, and hands focus on rather than dropping it', async () => {
    let localRuns: QueueLocalRun[] = [
      createTestLocalRun({ id: 'a', label: 'First run' }),
      createTestLocalRun({ id: 'b', label: 'Second run', projectId: 'project-2' }),
    ];
    const cancelLocalRun = vi.fn((projectId: string, runId: string) => {
      localRuns = localRuns.filter((run) => run.projectId !== projectId || run.id !== runId);
    });
    const rerender = () => render({ cancelLocalRun, localRuns });
    await rerender();

    await act(() => cancelFor('First run').click());
    await rerender();

    expect(cancelLocalRun).toHaveBeenCalledExactlyOnceWith('project-1', 'a');
    expect(host.querySelector('[role="alertdialog"]')).toBeNull();
    expect(rows()).toHaveLength(1);
    expect(document.activeElement).toBe(cancelFor('Second run'));

    await act(() => cancelFor('Second run').click());
    await rerender();

    expect(cancelLocalRun).toHaveBeenLastCalledWith('project-2', 'b');
    expect(section()).toBeNull();
    expect(document.activeElement).toBe(tab('All'));
  });

  it('hands focus to the previous Cancel when the last of several rows is canceled', async () => {
    let localRuns: QueueLocalRun[] = [
      createTestLocalRun({ id: 'a', label: 'First run' }),
      createTestLocalRun({ id: 'b', label: 'Second run' }),
      createTestLocalRun({ id: 'c', label: 'Third run' }),
    ];
    const cancelLocalRun = vi.fn((_projectId: string, runId: string) => {
      localRuns = localRuns.filter((run) => run.id !== runId);
    });
    await render({ cancelLocalRun, localRuns });

    await act(() => cancelFor('Third run').click());
    await render({ cancelLocalRun, localRuns });

    expect(cancelLocalRun).toHaveBeenCalledExactlyOnceWith('project-1', 'c');
    expect(document.activeElement).toBe(cancelFor('Second run'));
  });

  it('follows the queue scope setting: the active project alone, or every open project', async () => {
    const localRuns = [
      createTestLocalRun({ id: 'a', label: 'Here', projectId: 'project-1' }),
      createTestLocalRun({ id: 'b', label: 'Elsewhere', projectId: 'project-2' }),
    ];

    await render({ localRuns, queueJobsScope: 'active-project' });
    expect(rows().map((row) => row.querySelector('[data-list-primary]')?.textContent)).toEqual([
      expect.stringContaining('Here'),
    ]);
    expect(tab('All').textContent).toBe('All6');

    await render({ localRuns, queueJobsScope: 'all' });
    expect(rows()).toHaveLength(2);
    expect(tab('All').textContent).toBe('All7');
  });

  it('counts under All and Active, where it is listed, and steps aside for the finished filters', async () => {
    await render({ localRuns: [createTestLocalRun({ label: 'Fresh run' })] });

    expect(['All', 'Active', 'Done', 'Failed', 'Canceled'].map((label) => tab(label).textContent)).toEqual([
      'All6',
      'Active2',
      'Done2',
      'Failed1',
      'Canceled1',
    ]);

    for (const [label, listed] of [
      ['Done', false],
      ['Failed', false],
      ['Canceled', false],
      ['Active', true],
      ['All', true],
    ] as const) {
      await act(() => tab(label).click());
      expect(section() !== null, label).toBe(listed);
    }
  });
});
