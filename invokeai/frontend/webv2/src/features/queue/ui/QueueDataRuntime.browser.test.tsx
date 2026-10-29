/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const commands = vi.hoisted(() => ({ cancelScopedItems: vi.fn(() => Promise.resolve()) }));
vi.mock('@features/queue/publicApi', () => ({
  createProductionQueueRealtimeRuntime: () => ({ dispose: vi.fn(), start: vi.fn() }),
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
  queueCommands: commands,
}));
vi.mock('./queueDataStore', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueDataStore>()),
  refreshQueue: () => Promise.resolve(),
}));
vi.mock('@features/queue/data/generationDevicesStore', () => ({ refreshGenerationDevices: vi.fn() }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));

import type * as QueueDataStore from './queueDataStore';

import { requestQueueAction } from './queueActionRequests';
import { clearQueueConfirmation, requestQueueConfirmation } from './queueConfirmationStore';
import { QueueDataRuntime } from './QueueDataRuntime';
import { createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider, type QueueUiAdapter } from './QueueUiContext';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueueDataRuntime />
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  clearQueueConfirmation();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('animates a queue confirmation out with its text instead of unmounting it on close', async () => {
  await act(() =>
    requestQueueConfirmation({
      body: 'Cancel every item?',
      cancelLabel: 'Keep running',
      confirmLabel: 'Cancel all',
      onConfirm: vi.fn(),
      title: 'Clear',
    })
  );
  await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
  const dialog = document.querySelector('[role="alertdialog"]')!;
  // The way out names what it keeps, so it never reads like the cancel it declines.
  expect([...dialog.querySelectorAll('button')].map((button) => button.textContent)).toEqual(
    expect.arrayContaining(['Keep running', 'Cancel all'])
  );

  const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));

  // An unmounted dialog never reaches its closed state, so it cannot animate out; a retained one does, then leaves.
  expect(frames).not.toHaveLength(0);
  for (const frame of frames) {
    expect(frame.text).toContain('Cancel every item?');
  }
  await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
});

describe('a shortcut-requested cancel-all', () => {
  const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };

  const renderWithQueue = async (overrides: Partial<QueueUiAdapter> = {}) => {
    const client = new QueryClient();
    client.setQueryData(['queue'], {
      current: null,
      items: [],
      next: null,
      scope: {},
      status: {
        processor: { isProcessing: true, isStarted: true },
        queue: {
          canceled: 0,
          completed: 0,
          failed: 0,
          inProgress: 1,
          pending: 2,
          queueId: 'default',
          total: 3,
          waiting: 0,
        },
      },
    });
    const adapter = createTestQueueUiAdapter({
      activeProjectId: 'project-1',
      activeProjectName: 'Portraits',
      notify,
      queueJobsScope: 'active-project',
      ...overrides,
    });

    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <QueueUiProvider adapter={adapter}>
              <QueueDataRuntime />
            </QueueUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
    await act(() => requestQueueAction('cancel-all'));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');

    return document.querySelector<HTMLElement>('[role="alertdialog"]')!;
  };

  const buttonIn = (dialog: HTMLElement, label: string) =>
    [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === label)!;

  beforeEach(() => {
    commands.cancelScopedItems.mockClear();
    notify.success.mockClear();
  });

  it('confirms with the scope and count its menu item names', async () => {
    const dialog = await renderWithQueue();

    expect(dialog.textContent).toContain(
      'widgets.queue.cancelAllConfirmationBody.project {"count":3,"project":"Portraits"}'
    );
  });

  it('leaves the items alone when the way out, "Keep running", is chosen', async () => {
    const dialog = await renderWithQueue();

    await act(() => userEvent.click(buttonIn(dialog, 'widgets.queue.keepRunning')));

    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(commands.cancelScopedItems).not.toHaveBeenCalled();
  });

  it("stops this tab's runs the backend never got while offline, and sends nothing to it", async () => {
    const cancelLocalUnsubmittedRuns = vi.fn();
    const dialog = await renderWithQueue({
      cancelLocalUnsubmittedRuns,
      isConnected: false,
      localUnsubmittedRuns: { activeProject: 2, allProjects: 5 },
    });

    expect(dialog.textContent).toContain('"count":2');
    await act(() => userEvent.click(buttonIn(dialog, 'widgets.queue.cancelAllItems')));

    await vi.waitFor(() => expect(cancelLocalUnsubmittedRuns).toHaveBeenCalledWith('project-1'));
    expect(commands.cancelScopedItems).not.toHaveBeenCalled();
  });

  it('cancels the scope once confirmed', async () => {
    const dialog = await renderWithQueue();

    await act(() => userEvent.click(buttonIn(dialog, 'widgets.queue.cancelAllItems')));

    await vi.waitFor(() => expect(commands.cancelScopedItems).toHaveBeenCalledOnce());
    expect(commands.cancelScopedItems).toHaveBeenCalledWith(
      expect.objectContaining({ originPrefix: expect.any(String) })
    );
    await vi.waitFor(() => expect(notify.success).toHaveBeenCalled());
  });
});
