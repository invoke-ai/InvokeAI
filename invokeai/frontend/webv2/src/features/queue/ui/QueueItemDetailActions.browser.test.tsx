/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { QueueItemReadModel } from '@features/queue/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type * as QueueDataStore from './queueDataStore';

import { clearQueueConfirmation, getQueueConfirmation } from './queueConfirmationStore';
import { QueueItemDetailActions } from './QueueItemDetailActions';
import { QueueUiProvider, type QueueUiAdapter } from './QueueUiContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, values?: { count?: number }) => `${key}${values?.count ?? ''}` }),
}));
const commands = vi.hoisted(() => ({ cancelBatch: vi.fn<(batchId: string) => Promise<number>>() }));
vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'] }),
  queueCommands: commands,
}));
const refreshQueue = vi.hoisted(() => vi.fn(() => Promise.resolve()));
vi.mock('./queueDataStore', async (importOriginal) => ({
  ...(await importOriginal<typeof QueueDataStore>()),
  refreshQueue,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const item = { batchId: 'batch-7', id: 7, status: 'completed' } as QueueItemReadModel;
const notify = { error: vi.fn(), info: vi.fn(), success: vi.fn() };
const adapter = (canManage: boolean): QueueUiAdapter => ({
  ItemActions: () => null,
  activeProjectId: null,
  canManageInvocationCache: false,
  canManageItem: () => canManage,
  canManageProcessor: true,
  canViewItemDetails: () => true,
  isConnected: true,
  isProjectOpen: () => true,
  notify,
  openQueue: () => undefined,
  queueJobsScope: 'all',
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  clearQueueConfirmation();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = async (canManage = true) => {
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueueUiProvider adapter={adapter(canManage)}>
          <QueueItemDetailActions item={item} />
        </QueueUiProvider>
      </ChakraProvider>
    )
  );
};

const requestCancelBatch = async () => {
  await act(() => host.querySelector('button')!.click());
  return getQueueConfirmation();
};

describe('QueueItemDetailActions', () => {
  it('cancels the batch of a finished row only once the user confirms', async () => {
    commands.cancelBatch.mockResolvedValue(2);
    await render();

    expect((await requestCancelBatch())?.title).toBe('widgets.queue.cancelBatchTitle');
    clearQueueConfirmation();
    expect(commands.cancelBatch).not.toHaveBeenCalled();

    await act(() => requestCancelBatch().then((confirmation) => confirmation?.onConfirm()));

    expect(commands.cancelBatch).toHaveBeenCalledWith('batch-7');
    expect(notify.success).toHaveBeenCalledWith(
      'widgets.queue.batchCancellationRequested',
      'widgets.queue.batchCanceledCount2'
    );
  });

  it('says there was nothing left to cancel instead of reporting zero cancellations', async () => {
    commands.cancelBatch.mockResolvedValue(0);
    await render();

    await act(() => requestCancelBatch().then((confirmation) => confirmation?.onConfirm()));

    expect(notify.info).toHaveBeenCalledWith(
      'widgets.queue.batchNothingToCancel',
      'widgets.queue.batchNothingToCancelDescription'
    );
    expect(notify.success).not.toHaveBeenCalled();
  });

  it('reports a failed cancellation, but not to an account that signed in meanwhile', async () => {
    commands.cancelBatch.mockRejectedValueOnce(new Error('Server offline'));
    await render();

    await act(() => requestCancelBatch().then((confirmation) => confirmation?.onConfirm()));
    expect(notify.error).toHaveBeenCalledWith('widgets.queue.failedToCancelBatch', 'Server offline');

    let rejectCancel: (reason: Error) => void = () => undefined;
    commands.cancelBatch.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectCancel = reject;
        })
    );
    const confirmation = await requestCancelBatch();
    const pending = act(() => confirmation?.onConfirm());
    accountLifecycle.invalidate();
    rejectCancel(new Error('Aborted'));
    await pending;

    expect(notify.error).toHaveBeenCalledTimes(1);
  });

  it('reports nothing for a cancellation that finished after the account changed', async () => {
    let resolveCancel: (canceled: number) => void = () => undefined;
    commands.cancelBatch.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveCancel = resolve;
        })
    );
    await render();

    const confirmation = await requestCancelBatch();
    const pending = act(() => confirmation?.onConfirm());
    accountLifecycle.invalidate();
    resolveCancel(3);
    await pending;

    expect(refreshQueue).not.toHaveBeenCalled();
    expect(notify.success).not.toHaveBeenCalled();
    expect(notify.info).not.toHaveBeenCalled();
  });

  it('offers nothing to a viewer who cannot manage the item', async () => {
    await render(false);

    expect(host.querySelector('button')).toBeNull();
  });
});
