/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { QueueItemReadModel } from '@features/queue/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type * as QueueDataStore from './queueDataStore';

import { QueueItemRow } from './QueueItemRow';
import { QueueUiProvider, type QueueUiAdapter } from './QueueUiContext';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@features/queue/publicApi', () => ({
  getQueueReadModelOptions: () => ({ queryFn: () => Promise.resolve(null), queryKey: ['queue'] }),
  queueCommands: { cancelItem: () => Promise.resolve() },
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
  canManageItem: () => true,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  isConnected: true,
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

const render = async (ui: QueueUiAdapter, queueItem: QueueItemReadModel) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueueUiProvider adapter={ui}>
          <div role="list">
            <QueueItemRow item={queueItem} />
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
    expect(primary().textContent).toContain('A castle at dusk');

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

    await act(() => primary().click());
    expect(primary().getAttribute('aria-expanded')).toBe('false');
    expect(card.textContent).not.toContain('common.prompt');
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
});
