import type { AccountScope } from '@platform/state/accountLifecycle';

import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { QueueRunJournal } from './queueRunJournal';

const harness = vi.hoisted(() => {
  const controller = new AbortController();
  return {
    accountCurrent: true,
    deleteForProject: vi.fn(),
    downloadText: vi.fn(),
    journalClose: vi.fn(),
    listForProject: vi.fn(),
    listProjectIds: vi.fn(),
    lockResult: { kind: 'acquired', release: vi.fn(() => Promise.resolve()) } as
      | { kind: 'acquired'; release(): Promise<void> }
      | { kind: 'contended' }
      | { kind: 'unavailable' },
    onOpen: vi.fn(),
    owner: {
      accountId: 'account-1',
      epoch: 7,
      signal: controller.signal,
      storageSuffix: ':account-1',
    } as AccountScope,
  };
});

vi.mock('@platform/browser/downloadBlob', () => ({ downloadText: harness.downloadText }));
vi.mock('@platform/state/accountLifecycle', () => ({
  AccountScopeExpiredError: class AccountScopeExpiredError extends Error {},
  assertAccountScopeCurrent: () => {
    if (!harness.accountCurrent) {
      throw new Error('account expired');
    }
  },
  captureAccountScope: () => harness.owner,
  isAccountScopeCurrent: () => harness.accountCurrent,
}));
vi.mock('@platform/ui/ConfirmDialog', () => ({
  ConfirmDialog: ({ body, isOpen, onConfirm }: { body: string; isOpen: boolean; onConfirm(): Promise<void> | void }) =>
    isOpen ? (
      <div>
        <span>{body}</span>
        <button data-testid="confirm-discard" onClick={() => void onConfirm()}>
          confirm
        </button>
      </div>
    ) : null,
}));
vi.mock('@workbench/projects/projectLifecycleLocks', () => ({
  acquireProjectMutationLock: vi.fn(() => Promise.resolve(harness.lockResult)),
}));
vi.mock('./queueRunJournal', () => ({
  createAccountOwnedQueueRunJournal: vi.fn(() =>
    Promise.resolve({
      availability: 'available',
      close: harness.journalClose,
      deleteForProject: harness.deleteForProject,
      listForProject: harness.listForProject,
      listProjectIds: harness.listProjectIds,
    } as unknown as QueueRunJournal)
  ),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { count?: number; number?: number; projectId?: string }) =>
      values ? `${key}:${JSON.stringify(values)}` : key,
  }),
}));

import { QueueRecoveryNotice } from './QueueRecoveryNotice';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const EMPTY_OPEN_PROJECT_IDS: string[] = [];
const renderNotice = async ({ openProjectIds = EMPTY_OPEN_PROJECT_IDS }: { openProjectIds?: string[] } = {}) => {
  await act(() => {
    root?.render(
      <StrictMode>
        <QueryClientProvider client={queryClient!}>
          <ChakraProvider value={system}>
            <QueueRecoveryNotice openProjectIds={openProjectIds} onOpen={harness.onOpen} />
          </ChakraProvider>
        </QueryClientProvider>
      </StrictMode>
    );
  });
  await act(async () => {
    await vi.waitFor(() => expect(harness.listProjectIds).toHaveBeenCalled());
    await vi.waitFor(() => expect(document.body.textContent).not.toContain('shell.queueRecovery.loading'));
  });
};

const clickButton = async (text: string) => {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === text);
  expect(button).toBeDefined();
  await act(() => userEvent.click(button!));
};

beforeEach(() => {
  harness.accountCurrent = true;
  harness.deleteForProject.mockReset();
  harness.deleteForProject.mockResolvedValue({ kind: 'removed' });
  harness.downloadText.mockReset();
  harness.journalClose.mockReset();
  harness.listForProject.mockReset();
  harness.listForProject.mockResolvedValue({ entries: [], kind: 'available', removedCorrupt: 0 });
  harness.listProjectIds.mockReset();
  harness.listProjectIds.mockResolvedValue({
    kind: 'available',
    projectIds: ['open-project', 'closed-1', 'closed-2'],
  });
  harness.lockResult = { kind: 'acquired', release: vi.fn(() => Promise.resolve()) };
  harness.onOpen.mockReset();
  harness.onOpen.mockResolvedValue(undefined);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  host?.remove();
  host = null;
  queryClient = null;
  root = null;
});

describe('QueueRecoveryNotice', () => {
  it('lists only closed journal projects and exports the selected project in the explicit recovery format', async () => {
    harness.listForProject.mockImplementation((projectId) =>
      Promise.resolve({
        entries: [
          {
            item: { id: `run-${projectId}`, status: 'running' },
            projectId,
            queueItemId: `run-${projectId}`,
            submissionOrder: 3,
          },
        ],
        kind: 'available',
        removedCorrupt: 0,
      })
    );
    await renderNotice({ openProjectIds: ['open-project'] });

    expect(document.body.textContent).toContain('closed-1');
    expect(document.body.textContent).not.toContain('open-project');
    await clickButton('shell.queueRecovery.next');
    expect(document.body.textContent).toContain('closed-2');
    await clickButton('shell.queueRecovery.export');
    await vi.waitFor(() => expect(harness.downloadText).toHaveBeenCalledOnce());

    expect(harness.listForProject).toHaveBeenCalledWith('closed-2');
    const [json, fileName, mediaType] = harness.downloadText.mock.calls[0]!;
    expect(JSON.parse(json)).toEqual({
      exportedAt: expect.any(String),
      format: 'invokeai-webv2-queue-run-recovery',
      projectId: 'closed-2',
      runs: [
        {
          item: { id: 'run-closed-2', status: 'running' },
          queueItemId: 'run-closed-2',
          submissionOrder: 3,
        },
      ],
      schemaVersion: 1,
    });
    expect(fileName).toBe('closed-2.queue-recovery.json');
    expect(mediaType).toBe('application/json');
  });

  it('discards only the selected project after confirmation and releases the lifecycle lock', async () => {
    await renderNotice({ openProjectIds: ['open-project'] });
    await clickButton('shell.queueRecovery.next');
    await clickButton('shell.queueRecovery.discard');
    expect(document.body.textContent).toContain('shell.queueRecovery.discardConfirmBody');
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-discard"]')!));
    await vi.waitFor(() => expect(harness.deleteForProject).toHaveBeenCalledOnce());

    expect(harness.deleteForProject).toHaveBeenCalledWith('closed-2');
    expect(harness.lockResult.kind === 'acquired' ? harness.lockResult.release : undefined).toHaveBeenCalledOnce();
  });

  it('opens the selected existing project without reading its journal payload', async () => {
    await renderNotice({ openProjectIds: ['open-project'] });
    await clickButton('shell.queueRecovery.open');
    await vi.waitFor(() => expect(harness.onOpen).toHaveBeenCalledWith('closed-1'));

    expect(harness.listForProject).not.toHaveBeenCalled();
  });

  it('includes every journal project when no open-project filter is supplied', async () => {
    await renderNotice();
    await clickButton('shell.queueRecovery.next');
    await clickButton('shell.queueRecovery.next');

    expect(document.body.textContent).toContain('open-project');
  });

  it('fails closed when another tab holds the project lifecycle lock', async () => {
    harness.lockResult = { kind: 'contended' };
    await renderNotice({ openProjectIds: ['open-project'] });
    await clickButton('shell.queueRecovery.discard');
    await act(() => userEvent.click(document.querySelector<HTMLButtonElement>('[data-testid="confirm-discard"]')!));
    await vi.waitFor(() => expect(document.body.textContent).toContain('shell.queueRecovery.lockContended'));

    expect(harness.deleteForProject).not.toHaveBeenCalled();
  });

  it('shows unknown recovery state when IndexedDB cannot be inspected', async () => {
    harness.listProjectIds.mockResolvedValueOnce({ kind: 'unavailable' });
    await renderNotice();

    expect(document.body.textContent).toContain('shell.queueRecovery.storageUnavailable');
    expect(document.body.textContent).not.toContain('shell.queueRecovery.none');
  });

  it('suppresses downloads and state updates when the account or component expires during export', async () => {
    let resolveLoad: ((value: { entries: []; kind: 'available'; removedCorrupt: 0 }) => void) | null = null;
    harness.listForProject.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveLoad = resolve;
        })
    );
    await renderNotice({ openProjectIds: ['open-project'] });
    await clickButton('shell.queueRecovery.export');
    const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>('button'));
    expect(buttons.every((button) => button.disabled)).toBe(true);

    harness.accountCurrent = false;
    await act(() => root?.unmount());
    await act(() => resolveLoad?.({ entries: [], kind: 'available', removedCorrupt: 0 }));

    expect(harness.downloadText).not.toHaveBeenCalled();
  });
});
