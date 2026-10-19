/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { QueueItemReadModel } from '@features/queue/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { buildQueueItemOrigin } from '@features/queue/data/events';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type * as QueueDataStore from './queueDataStore';

import { QueueItemRow } from './QueueItemRow';
import { QueueUiProvider, type QueueUiAdapter } from './QueueUiContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { id?: number }) =>
      key === 'widgets.queue.itemTitle' ? `Queue item #${values?.id}` : key,
  }),
}));
const commands = vi.hoisted(() => ({
  cancelItem: vi.fn(() => Promise.resolve()),
  retryItems: vi.fn((itemIds: number[]) => Promise.resolve(itemIds)),
}));

vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({
    queryFn: () => Promise.resolve(null),
    queryKey: ['queue', 'item-detail', itemId],
  }),
  getQueueReadModelOptions: () => ({ queryFn: () => Promise.resolve(null), queryKey: ['queue'] }),
  queueCommands: commands,
}));
vi.mock('./queueDataStore', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueDataStore>()),
  refreshQueue: () => Promise.resolve(),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const item = (overrides: Partial<QueueItemReadModel> = {}): QueueItemReadModel => ({
  batchId: 'batch-1',
  createdAt: '2026-06-27T00:00:00Z',
  fieldValues: [{ fieldName: 'prompt', nodePath: 'positive_prompt', value: 'A castle at dusk' }],
  id: 7,
  resultImageNames: [],
  sessionId: 'session-1',
  status: 'pending',
  updatedAt: '2026-06-27T00:00:00Z',
  userId: 'user-1',
  ...overrides,
});

const adapter = (overrides: Partial<QueueUiAdapter> = {}): QueueUiAdapter => ({
  ItemActions: () => null,
  activeProjectId: null,
  canManageInvocationCache: false,
  canManageItem: () => true,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  isConnected: true,
  isProjectOpen: () => true,
  notify: { error: () => undefined, info: () => undefined, success: () => undefined },
  openQueue: () => undefined,
  preloadItemActions: () => undefined,
  queueJobsScope: 'all',
  ...overrides,
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'width:420px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async (ui: QueueUiAdapter, ...queueItems: QueueItemReadModel[]) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueueUiProvider adapter={ui}>
          <div role="list">
            {queueItems.map((queueItem) => (
              <QueueItemRow key={queueItem.id} item={queueItem} />
            ))}
          </div>
        </QueueUiProvider>
      </ChakraProvider>
    )
  );
};

const primary = () => host.querySelector<HTMLElement>('[data-list-primary]')!;

describe('QueueItemRow', () => {
  it('is a list item whose row button discloses details while the cancel control stays beside it', async () => {
    await render(adapter(), item());

    const card = host.querySelector<HTMLElement>('[role="list"] > [role="listitem"]')!;

    expect(card).not.toBeNull();
    expect(card.querySelector('[role="presentation"] [data-list-primary]')).toBe(primary());
    expect(primary().tagName).toBe('BUTTON');
    expect(primary().getAttribute('aria-expanded')).toBe('false');
    expect(primary().textContent).toContain('Queue item #7');
    expect(card.textContent).not.toContain('A castle at dusk');

    const cancel = host.querySelector<HTMLButtonElement>('button[aria-label="widgets.queue.cancelItem"]')!;

    expect(cancel).not.toBeNull();
    expect(primary().contains(cancel)).toBe(false);
    expect(card.contains(cancel)).toBe(true);

    await act(() => cancel.click());
    expect(primary().getAttribute('aria-expanded')).toBe('false');

    await act(() => primary().click());
    expect(primary().getAttribute('aria-expanded')).toBe('true');
    // The details stay inside the list item, so they read as part of the row.
    expect(card.textContent).toContain('common.prompt');
    expect(card.textContent).toContain('A castle at dusk');
    expect(primary().textContent).toContain('Queue item #7');

    await act(() => primary().click());
    expect(primary().getAttribute('aria-expanded')).toBe('false');
    expect(card.textContent).not.toContain('common.prompt');
    expect(card.textContent).not.toContain('A castle at dusk');
  });

  it('distinguishes workflow rows without prompt metadata by their queue item ids', async () => {
    await render(adapter(), item({ fieldValues: [], id: 21 }), item({ fieldValues: [], id: 22 }));

    const rows = host.querySelectorAll<HTMLElement>('[data-list-primary]');

    expect(rows).toHaveLength(2);
    expect(rows[0]!.textContent).toContain('Queue item #21');
    expect(rows[1]!.textContent).toContain('Queue item #22');
    expect(host.textContent).not.toContain('widgets.queue.noPrompt');
  });

  it('renders a row without viewable details as static content, not a button', async () => {
    await render(
      adapter({ canManageItem: () => false, canViewItemDetails: () => false }),
      item({ userId: 'redacted' })
    );

    expect(primary().tagName).toBe('DIV');
    expect(primary().hasAttribute('aria-expanded')).toBe(false);
    expect(host.querySelector('button')).toBeNull();
    expect(host.querySelector('[role="list"] > [role="listitem"]')).not.toBeNull();
  });

  it('warms the item actions on focus, before the first expand', async () => {
    const preloadItemActions = vi.fn();

    await render(adapter({ preloadItemActions }), item());
    // Dispatch focusin directly: element.focus() is a no-op for focus events while the test window is unfocused.
    await act(() => {
      primary().dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    });

    expect(preloadItemActions).toHaveBeenCalledTimes(1);
  });

  it('lets a waiting workflow-call parent be canceled from its own row', async () => {
    commands.cancelItem.mockClear();
    await render(adapter(), item({ id: 11, status: 'waiting' }));

    const cancel = host.querySelector<HTMLButtonElement>('button[aria-label="widgets.queue.cancelItem"]')!;

    expect(cancel).not.toBeNull();
    expect(host.textContent).toContain('common.status.waiting');
    await act(() => cancel.click());
    expect(commands.cancelItem).toHaveBeenCalledWith(11);
  });

  it('offers retry on a failed root item and reports the retried id', async () => {
    const success = vi.fn();
    commands.retryItems.mockClear();
    await render(
      adapter({ notify: { error: () => undefined, info: () => undefined, success } }),
      item({ id: 12, status: 'failed' })
    );

    expect(host.querySelector('button[aria-label="widgets.queue.cancelItem"]')).toBeNull();
    const retry = host.querySelector<HTMLButtonElement>('button[aria-label="widgets.queue.retryItem"]')!;

    expect(retry).not.toBeNull();
    await act(async () => {
      retry.click();
      await Promise.resolve();
    });
    expect(commands.retryItems).toHaveBeenCalledWith([12]);
    expect(success).toHaveBeenCalledWith('widgets.queue.retryQueued', 'widgets.queue.retryQueuedDescription');
  });

  it('retries a Canvas job of a project closed here, and says another tab with it open receives the result', async () => {
    const success = vi.fn();
    const canvasJob = (id: number, projectId: string) =>
      item({ destination: 'canvas', id, origin: buildQueueItemOrigin('run-1', projectId), status: 'failed' });
    const retry = async (queueItem: QueueItemReadModel) => {
      await act(() => root.unmount());
      root = createRoot(host);
      await render(
        adapter({
          isProjectOpen: (projectId) => projectId === 'open-project',
          notify: { error: () => undefined, info: () => undefined, success },
        }),
        queueItem
      );
      await act(async () => {
        host.querySelector<HTMLButtonElement>('button[aria-label="widgets.queue.retryItem"]')!.click();
        await Promise.resolve();
      });
    };
    commands.retryItems.mockClear();

    await retry(canvasJob(15, 'closed-project'));
    expect(commands.retryItems).toHaveBeenCalledWith([15]);
    expect(success).toHaveBeenLastCalledWith(
      'widgets.queue.retryQueued',
      'widgets.queue.retryQueuedProjectClosedDescription'
    );

    await retry(canvasJob(16, 'open-project'));
    expect(success).toHaveBeenLastCalledWith('widgets.queue.retryQueued', 'widgets.queue.retryQueuedDescription');
  });

  it('hides retry on a failed workflow-call child and on rows the viewer cannot manage', async () => {
    await render(adapter(), item({ id: 13, parentItemId: 5, status: 'failed' }));
    expect(host.querySelector('button[aria-label="widgets.queue.retryItem"]')).toBeNull();

    await act(() => root.unmount());
    root = createRoot(host);
    await render(adapter({ canManageItem: () => false }), item({ id: 14, status: 'canceled' }));
    expect(host.querySelector('button[aria-label="widgets.queue.retryItem"]')).toBeNull();
  });
});
