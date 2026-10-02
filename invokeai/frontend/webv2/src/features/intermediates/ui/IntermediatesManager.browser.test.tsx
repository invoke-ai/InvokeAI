import type {
  IntermediatesOperation,
  IntermediatesPreview,
  IntermediatesRow,
  IntermediatesSummary,
} from '@features/intermediates/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

const dependencies = vi.hoisted(() => ({
  createIntermediatesPreview: vi.fn(),
  getIntermediatesOperation: vi.fn(),
  getIntermediatesSummary: vi.fn(),
  listIntermediatesOperations: vi.fn(),
  startIntermediatesOperation: vi.fn(),
}));

vi.mock('@features/intermediates/data/api', () => ({
  createIntermediatesPreview: dependencies.createIntermediatesPreview,
  getIntermediatesOperation: dependencies.getIntermediatesOperation,
  getIntermediatesSummary: dependencies.getIntermediatesSummary,
  listIntermediatesOperations: dependencies.listIntermediatesOperations,
  startIntermediatesOperation: dependencies.startIntermediatesOperation,
}));
vi.mock('@features/intermediates/data/realtime', () => ({ attachIntermediatesRealtime: () => () => undefined }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { resolvedLanguage: 'en' },
    t: (key: string, options?: Record<string, unknown>) =>
      options && Object.keys(options).length > 0
        ? `${key}(${Object.entries(options)
            .map(([name, value]) => `${name}=${String(value)}`)
            .join(',')})`
        : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const row = (projectId: string | null, name: string | null, safe: number): IntermediatesRow => ({
  coverImageName: null,
  images: { active: 1, recent: 0, referenced: 2, safe },
  projectId,
  projectName: name,
  reclaimableBytes: safe * 1_048_576,
  referencedBytes: 0,
  unknownSizeCount: 0,
  userDisplayName: 'Alice',
  userEmail: 'alice@example.com',
  userId: 'alice',
  videos: { active: 0, recent: 0, referenced: 0, safe: 0 },
});

const summaryOf = (items: IntermediatesRow[]): IntermediatesSummary => ({
  canManageEveryone: false,
  items,
  limit: 50,
  measuring: false,
  offset: 0,
  recentGraceSeconds: 1800,
  total: items.length,
  totals: {
    inUseImages: items.length * 3,
    inUseVideos: 0,
    reclaimableBytes: items.reduce((sum, item) => sum + item.reclaimableBytes, 0),
    rows: items.length,
    safeImages: items.reduce((sum, item) => sum + item.images.safe, 0),
    safeVideos: 0,
    unknownSizeCount: 0,
  },
});

const documentOf = (userId: string) => ({
  userDisplayName: userId === 'bob' ? 'Bob' : 'Alice',
  userEmail: `${userId}@example.com`,
  userId,
});

const forceAffectedDocuments: IntermediatesPreview['affectedDocuments'] = [
  { ...documentOf('alice'), kind: 'project', name: 'Portraits', ownerId: 'p1', references: 2 },
  { ...documentOf('alice'), kind: 'client_state', name: null, ownerId: 'canvas', references: 1 },
  { ...documentOf('alice'), kind: 'quarantined_project', name: 'Old sketch', ownerId: 'q1', references: 1 },
  { ...documentOf('bob'), kind: 'client_state', name: null, ownerId: 'canvas', references: 1 },
];

const previewOf = (mode: 'safe' | 'force', deleteImages: number): IntermediatesPreview => ({
  affectedDocuments: mode === 'force' ? forceAffectedDocuments : [],
  affectedDocumentsTotal: mode === 'force' ? forceAffectedDocuments.length : 0,
  createdAt: '2026-09-24T12:00:00.000Z',
  expiresAt: '2026-09-24T12:10:00.000Z',
  impact: {
    deleteImages,
    deleteVideos: 0,
    keepActiveImages: 1,
    keepActiveVideos: 0,
    keepRecentImages: 0,
    keepRecentVideos: 0,
    keepReferencedImages: mode === 'safe' ? 2 : 0,
    keepReferencedVideos: 0,
    reclaimableBytes: deleteImages * 1_048_576,
    unknownSizeCount: 0,
  },
  mode,
  previewId: `preview-${mode}`,
  scope: { kind: 'owner', userId: 'alice' },
  targetRows: 1,
});

const operationOf = (status: IntermediatesOperation['status']): IntermediatesOperation => ({
  completedAt: status === 'completed' ? 'later' : null,
  createdAt: '2026-09-24T12:00:00.000Z',
  error: null,
  mode: 'safe',
  operationId: 'op-1',
  progress: {
    deletedImages: status === 'completed' ? 4 : 2,
    deletedVideos: 0,
    failedImages: 0,
    failedVideos: 0,
    pendingDiskCleanup: 0,
    processedImages: status === 'completed' ? 4 : 2,
    processedVideos: 0,
    reclaimedBytes: 4_194_304,
    retainedImages: 0,
    retainedVideos: 0,
    unknownSizeCount: 0,
  },
  scope: { kind: 'owner', userId: 'alice' },
  startedAt: 'now',
  status,
  targetImages: 4,
  targetVideos: 0,
  userId: 'alice',
});

const matchingScope = (overrides: Partial<Extract<IntermediatesOperation['scope'], { kind: 'matching' }>> = {}) => ({
  excluded: [],
  kind: 'matching' as const,
  projectId: null,
  search: null,
  userId: 'alice',
  ...overrides,
});

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;

const renderManager = async (
  options: {
    focusProjectId?: string;
    focusOwner?: { ownerId: string; ownerLabel?: string };
    currentUserId?: string | null;
    currentUserLabel?: string;
    canClearOthersIntermediates?: boolean;
    strict?: boolean;
  } = {}
): Promise<void> => {
  host = document.createElement('div');
  host.style.height = '640px';
  host.style.width = '900px';
  host.style.display = 'flex';
  host.style.flexDirection = 'column';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { requestIntermediatesFocus } = await import('@features/intermediates/data/focus');
  const { IntermediatesManager } = await import('./IntermediatesManager');

  if (options.focusProjectId || options.focusOwner) {
    requestIntermediatesFocus({ projectId: options.focusProjectId, ...options.focusOwner });
  }
  const manager = (
    <ChakraProvider value={system}>
      <QueryClientProvider client={queryClient}>
        <IntermediatesManager
          canClearOthersIntermediates={options.canClearOthersIntermediates ?? false}
          currentUserId={options.currentUserId === undefined ? 'alice' : options.currentUserId}
          currentUserLabel={options.currentUserLabel}
        />
      </QueryClientProvider>
    </ChakraProvider>
  );
  await act(() => {
    root.render(options.strict ? <StrictMode>{manager}</StrictMode> : manager);
  });
};

/** Chakra puts the accessible name on the checkbox root; the hidden input inside it carries the state. */
const checkbox = (label: string): { click: () => void; checked: boolean } => {
  const rootElement = host.querySelector<HTMLElement>(`[aria-label="${label}"]`);
  expect(rootElement, label).not.toBeNull();
  const input = rootElement!.querySelector<HTMLInputElement>('input');
  expect(input, `${label} input`).not.toBeNull();
  return { checked: input!.checked, click: () => input!.click() };
};

const buttonWithText = (text: string, scope: ParentNode = document): HTMLButtonElement => {
  const button = [...scope.querySelectorAll<HTMLButtonElement>('button')].find((candidate) =>
    candidate.textContent?.includes(text)
  );
  expect(button, text).toBeDefined();
  return button!;
};

/** Serves `operation` to polls and pushes it into the cache the way a realtime update does, without waiting a poll. */
const followOperation = async (initial: IntermediatesOperation) => {
  const { intermediatesOperationQueryOptions } = await import('@features/intermediates/data/queries');
  let current = initial;
  dependencies.getIntermediatesOperation.mockImplementation(() => Promise.resolve(current));
  return async (next: IntermediatesOperation) => {
    current = next;
    await act(() => {
      queryClient.setQueryData(intermediatesOperationQueryOptions(next.operationId).queryKey, next);
    });
  };
};

/** Delete stays focusable while unavailable, so the reason it waits can be announced. */
const isDeleteUnavailable = (): boolean =>
  buttonWithText('intermediates.list.delete', host).getAttribute('aria-disabled') === 'true';

const setInputValue = (input: HTMLInputElement, value: string) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  setter.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

const openDialog = async (): Promise<HTMLElement> => {
  await act(() => buttonWithText('intermediates.list.delete', host).click());
  const dialog = await vi.waitFor(() => {
    const element = document.querySelector<HTMLElement>('[role="alertdialog"]');
    expect(element).not.toBeNull();
    return element!;
  });
  await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.reclaim'));
  return dialog;
};

const openForceDialogReadyToConfirm = async (
  options: { canClearOthersIntermediates?: boolean } = {}
): Promise<HTMLElement> => {
  await renderManager({ focusProjectId: 'p1', ...options });
  await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
  const dialog = await openDialog();
  const toggles = () => [...dialog.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
  await act(() => toggles()[0]!.click());
  await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.affectedClientState'));
  await act(() => toggles()[1]!.click());
  await act(() => setInputValue(dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])')!, 'DELETE'));
  return dialog;
};

const INTERMEDIATES_FIELD = { id: 'intermediatesManager', kind: 'custom', label: '', scope: 'none' } as const;

describe('IntermediatesManager', () => {
  beforeEach(() => {
    dependencies.getIntermediatesSummary
      .mockReset()
      .mockImplementation((params: { search?: string }) =>
        Promise.resolve(
          summaryOf(
            params.search?.trim()
              ? [row('p1', 'Portraits', 4)]
              : [row('p1', 'Portraits', 4), row('p2', 'Landscapes', 1), row(null, null, 7)]
          )
        )
      );
    dependencies.createIntermediatesPreview
      .mockReset()
      .mockImplementation(({ mode }: { mode: 'safe' | 'force' }) => Promise.resolve(previewOf(mode, 4)));
    dependencies.startIntermediatesOperation.mockReset().mockResolvedValue(operationOf('running'));
    dependencies.getIntermediatesOperation.mockReset().mockResolvedValue(operationOf('completed'));
    dependencies.listIntermediatesOperations.mockReset().mockResolvedValue([]);
  });

  afterEach(async () => {
    // A confirmation torn down while open restores focus later, into the next test; close it first.
    const cancel = document.querySelector<HTMLButtonElement>('[role="alertdialog"] button:not([disabled])');
    if (cancel && cancel.textContent?.includes('common.cancel')) {
      await act(() => cancel.click());
      await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
    }
    await act(() => root?.unmount());
    host?.remove();
    const { followIntermediatesOperation } = await import('@features/intermediates/data/operationStore');
    followIntermediatesOperation(null);
  });

  it('starts an admin in their own account and exposes an explicit everyone switch', async () => {
    await renderManager({ canClearOthersIntermediates: true });
    host.style.width = '400px';
    await vi.waitFor(() =>
      expect(dependencies.getIntermediatesSummary).toHaveBeenCalledWith(
        expect.objectContaining({ ownerId: 'alice' }),
        expect.anything()
      )
    );
    const everyoneControl = host.querySelector<HTMLElement>('[aria-label="intermediates.owner.showEveryone"]')!;
    expect(everyoneControl.getBoundingClientRect().right).toBeLessThanOrEqual(host.getBoundingClientRect().right);
    await page.screenshot({ path: '../../../../artifacts/intermediates/admin-own-narrow.png' });
    await act(() => host.querySelector<HTMLButtonElement>('[aria-label="intermediates.owner.showEveryone"]')!.click());
    await vi.waitFor(() =>
      expect(dependencies.getIntermediatesSummary).toHaveBeenLastCalledWith(
        expect.objectContaining({ ownerId: null }),
        expect.anything()
      )
    );
    // Rows of every account name their owner, so assistive tech can tell identical project names apart.
    await vi.waitFor(() =>
      expect(host.querySelector('[aria-label^="intermediates.list.selectRowForOwner("]')).not.toBeNull()
    );
    await act(() => buttonWithText('intermediates.owner.showMine', host).click());
    expect(host.querySelector('[aria-label="intermediates.owner.showEveryone"]')).not.toBeNull();
    expect(host.textContent).not.toContain('intermediates.owner.showMine');
  });

  it('fetches a focused project by id even when it is outside the first page', async () => {
    dependencies.getIntermediatesSummary.mockImplementation((params: { projectId?: string }) =>
      Promise.resolve(summaryOf(params.projectId === 'p55' ? [row('p55', 'Far project', 2)] : [row('p1', 'First', 1)]))
    );
    await renderManager({ focusProjectId: 'p55' });
    await vi.waitFor(() => expect(host.textContent).toContain('Far project'));
    expect(checkbox('intermediates.list.selectRow(name=Far project)').checked).toBe(true);
    expect(dependencies.getIntermediatesSummary).toHaveBeenCalledWith(
      expect.objectContaining({ projectId: 'p55' }),
      expect.anything()
    );
  });

  it('keeps Select all within a focused project', async () => {
    dependencies.getIntermediatesSummary.mockImplementation((params: { projectId?: string }) =>
      Promise.resolve(summaryOf(params.projectId === 'p1' ? [row('p1', 'Portraits', 4)] : [row('p2', 'Other', 3)]))
    );
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    // The focused project is the whole filtered list, so the first click clears; the second selects all matching.
    await act(() => checkbox('intermediates.list.selectAll').click());
    expect(isDeleteUnavailable()).toBe(true);
    await act(() => checkbox('intermediates.list.selectAll').click());
    await act(() => buttonWithText('intermediates.list.delete', host).click());
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenCalledWith(
        { mode: 'safe', scope: matchingScope({ projectId: 'p1' }) },
        expect.anything()
      )
    );
  });

  it('moves focus to the started operation, whose status the Delete control no longer reflects', async () => {
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    const dialog = await openDialog();
    await act(() => buttonWithText('intermediates.dialog.confirm', dialog).click());
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
    await vi.waitFor(
      () => {
        const focused = document.activeElement as HTMLElement;
        expect(focused.getAttribute('tabindex')).toBe('-1');
        expect(focused.textContent).toContain('intermediates.operation.status.running');
      },
      { timeout: 3_000 }
    );
  });

  it('re-follows a running operation after a reload by asking the server', async () => {
    const { activeOperationStore } = await import('@features/intermediates/data/operationStore');
    const running = { ...operationOf('running'), operationId: 'op-running' };
    dependencies.listIntermediatesOperations.mockResolvedValue([
      running,
      { ...operationOf('completed'), operationId: 'op-0' },
    ]);
    dependencies.getIntermediatesOperation.mockResolvedValue(running);

    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.status.running'));
    expect(activeOperationStore.getSnapshot().operationId).toBe('op-running');
    expect(dependencies.listIntermediatesOperations).toHaveBeenCalledOnce();
    expect(dependencies.startIntermediatesOperation).not.toHaveBeenCalled();
  });

  it('does not let a late operations listing replace a newer confirmed operation', async () => {
    const { activeOperationStore } = await import('@features/intermediates/data/operationStore');
    let resolveListing!: (operations: IntermediatesOperation[]) => void;
    dependencies.listIntermediatesOperations.mockImplementation(
      () =>
        new Promise<IntermediatesOperation[]>((resolve) => {
          resolveListing = resolve;
        })
    );
    dependencies.startIntermediatesOperation.mockResolvedValue({ ...operationOf('running'), operationId: 'new-op' });
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(dependencies.listIntermediatesOperations).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    const dialog = await openDialog();
    await act(() => buttonWithText('intermediates.dialog.confirm', dialog).click());
    await vi.waitFor(() => expect(activeOperationStore.getSnapshot().operationId).toBe('new-op'));

    await act(() => resolveListing([{ ...operationOf('running'), operationId: 'old-op' }]));
    expect(activeOperationStore.getSnapshot().operationId).toBe('new-op');
  });

  it('re-enables Confirm and follows nothing when sign-out cuts a start off', async () => {
    const { activeOperationStore } = await import('@features/intermediates/data/operationStore');
    accountLifecycle.activate('alice');
    try {
      dependencies.startIntermediatesOperation.mockImplementation(
        (_request: unknown, signal: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
          })
      );
      const dialog = await openForceDialogReadyToConfirm();
      await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());
      await vi.waitFor(() => expect(buttonWithText('intermediates.dialog.forceConfirm', dialog).disabled).toBe(true));

      await act(() => {
        accountLifecycle.activate('mallory');
      });
      await vi.waitFor(() => expect(buttonWithText('intermediates.dialog.forceConfirm', dialog).disabled).toBe(false));
      expect(activeOperationStore.getSnapshot().operationId).toBeNull();
    } finally {
      accountLifecycle.invalidate();
    }
  });

  it.each(['running', 'completed'] as const)(
    'follows the %s run the server lists when a start timed out',
    async (status) => {
      // The mount listing answers before the start; a small cleanup may even finish before the fresh listing.
      const accepted = { ...operationOf(status), operationId: 'op-accepted' };
      dependencies.startIntermediatesOperation.mockRejectedValueOnce(new DOMException('timed out', 'TimeoutError'));
      dependencies.listIntermediatesOperations.mockImplementation(() =>
        Promise.resolve(dependencies.startIntermediatesOperation.mock.calls.length > 0 ? [accepted] : [])
      );
      dependencies.getIntermediatesOperation.mockResolvedValue(accepted);
      const dialog = await openForceDialogReadyToConfirm();

      await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());

      await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
      await vi.waitFor(() => expect(host.textContent).toContain(`intermediates.operation.status.${status}`));
      expect(dependencies.startIntermediatesOperation).toHaveBeenCalledOnce();
      await vi.waitFor(() => expect((document.activeElement as HTMLElement).getAttribute('tabindex')).toBe('-1'), {
        timeout: 3_000,
      });
    }
  );

  it('explains a timed-out start that the server does not list and lets Confirm try again', async () => {
    dependencies.startIntermediatesOperation.mockRejectedValueOnce(new DOMException('timed out', 'TimeoutError'));
    const dialog = await openForceDialogReadyToConfirm();
    await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());
    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.startTimedOut'));
    const confirm = buttonWithText('intermediates.dialog.forceConfirm', dialog);
    expect(confirm.disabled).toBe(false);

    await act(() => confirm.click());
    await vi.waitFor(() => expect(dependencies.startIntermediatesOperation).toHaveBeenCalledTimes(2));
    const [first, second] = dependencies.startIntermediatesOperation.mock.calls;
    expect(second![0]).toEqual(first![0]);
    expect(first![0]).toEqual({ previewId: 'preview-force' });
  });

  it('never mistakes an earlier finished run for a timed-out start', async () => {
    const { activeOperationStore } = await import('@features/intermediates/data/operationStore');
    // Retained on the server from before this preview (its created_at is older than the preview's).
    const earlier = { ...operationOf('completed'), createdAt: '2026-09-24T11:00:00.000Z', operationId: 'op-earlier' };
    dependencies.listIntermediatesOperations.mockResolvedValue([earlier]);
    dependencies.startIntermediatesOperation.mockRejectedValueOnce(new DOMException('timed out', 'TimeoutError'));
    const dialog = await openForceDialogReadyToConfirm();
    // The catch-up on open ignores a settled run; Dismiss-then-Confirm leaves nothing followed.
    expect(activeOperationStore.getSnapshot().operationId).toBeNull();

    await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());

    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.startTimedOut'));
    expect(activeOperationStore.getSnapshot().operationId).toBeNull();
    expect(host.textContent).not.toContain('intermediates.operation.status.completed');
  });

  it('asks the server afresh after a timed-out start, and says so when that fails too', async () => {
    let resolveMountListing!: (operations: IntermediatesOperation[]) => void;
    dependencies.listIntermediatesOperations
      .mockImplementationOnce(
        () =>
          new Promise<IntermediatesOperation[]>((resolve) => {
            resolveMountListing = resolve;
          })
      )
      .mockRejectedValue(new ApiError('Gateway timeout', 504));
    dependencies.startIntermediatesOperation.mockRejectedValueOnce(new DOMException('timed out', 'TimeoutError'));
    const dialog = await openForceDialogReadyToConfirm();
    await vi.waitFor(() => expect(dependencies.listIntermediatesOperations).toHaveBeenCalledOnce());

    await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());

    // The listing begun at mount is still pending and could not know about this start: a second one is made.
    await vi.waitFor(() => expect(dependencies.listIntermediatesOperations).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.startUnknown'));
    expect(dialog.textContent).not.toContain('intermediates.dialog.startTimedOut');
    expect(buttonWithText('common.cancel', dialog).disabled).toBe(false);
    await act(() => resolveMountListing([]));
  });

  it('offers a fresh check when the confirmed preview is no longer known', async () => {
    dependencies.startIntermediatesOperation.mockRejectedValueOnce(new ApiError('Preview expired or unknown', 404));
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    const dialog = await openDialog();

    await act(() => buttonWithText('intermediates.dialog.confirm', dialog).click());

    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.previewExpired'));
    expect(buttonWithText('intermediates.dialog.confirm', dialog).disabled).toBe(true);
    await act(() => buttonWithText('common.retry', dialog).click());
    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.reclaim'));
    expect(buttonWithText('intermediates.dialog.confirm', dialog).disabled).toBe(false);
    expect(dependencies.createIntermediatesPreview).toHaveBeenCalledTimes(2);
  });

  it('reports a failed start in the dialog and follows nothing', async () => {
    const { activeOperationStore } = await import('@features/intermediates/data/operationStore');
    const dialog = await openForceDialogReadyToConfirm();
    expect(dialog.textContent).toContain('intermediates.dialog.affectedQuarantinedProject(count=1,name=Old sketch)');
    expect(dialog.textContent).toContain('intermediates.dialog.forceWarningOwn');
    dependencies.startIntermediatesOperation.mockRejectedValueOnce(new ApiError('Server error', 500));

    await act(() => buttonWithText('intermediates.dialog.forceConfirm', dialog).click());

    await vi.waitFor(() => expect(dialog.textContent).toContain('Server error'));
    expect(buttonWithText('intermediates.dialog.forceConfirm', dialog).disabled).toBe(false);
    expect(activeOperationStore.getSnapshot().operationId).toBeNull();
  });

  it('lists projects with used and unused counts and keeps Delete disabled until something is selected', async () => {
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));

    expect(host.textContent).toContain('intermediates.list.unassigned');
    expect(host.querySelectorAll('[role="list"] [role="listitem"]')).toHaveLength(3);
    expect(host.textContent).toContain('intermediates.list.used');
    expect(host.textContent).toContain('intermediates.list.unused');
    expect(isDeleteUnavailable()).toBe(true);
    await act(() => buttonWithText('intermediates.list.delete', host).click());
    expect(dependencies.createIntermediatesPreview).not.toHaveBeenCalled();

    await act(() => checkbox('intermediates.list.selectRow(name=Portraits)').click());
    expect(isDeleteUnavailable()).toBe(false);
    expect(host.textContent).toContain('count=1');

    await act(() => checkbox('intermediates.list.selectAll').click());
    expect(host.textContent).toContain('count=3');

    const search = host.querySelector<HTMLInputElement>('input[aria-label="intermediates.searchLabel"]');
    expect(search).not.toBeNull();
    await act(() => setInputValue(search!, 'Port'));
    await vi.waitFor(() =>
      expect(dependencies.getIntermediatesSummary).toHaveBeenLastCalledWith(
        expect.objectContaining({ search: 'Port' }),
        expect.anything()
      )
    );
    // A search hides rows; hidden selections must not survive it.
    expect(isDeleteUnavailable()).toBe(true);
  });

  it('previews the delete, starts the operation, and reports it once it settles', async () => {
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    expect(checkbox('intermediates.list.selectRow(name=Portraits)').checked).toBe(true);

    await act(() => buttonWithText('intermediates.list.delete', host).click());
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenCalledWith(
        { mode: 'safe', scope: { kind: 'selection', targets: [{ projectId: 'p1', userId: 'alice' }] } },
        expect.anything()
      )
    );
    const dialog = document.querySelector('[role="alertdialog"]');
    expect(dialog).not.toBeNull();
    await vi.waitFor(() => expect(dialog!.textContent).toContain('intermediates.dialog.reclaim(size=4.2 MB)'));
    expect(dialog!.textContent).toContain('intermediates.dialog.kept(count=3)');
    // The impact is the dialog's accessible description even though it arrives after the dialog opens.
    const describedBy = dialog!.getAttribute('aria-describedby');
    expect(describedBy).not.toBeNull();
    expect(document.getElementById(describedBy!)?.textContent).toContain('intermediates.dialog.reclaim(size=4.2 MB)');

    const confirm = buttonWithText('intermediates.dialog.confirm', dialog!);
    expect(confirm.disabled).toBe(false);
    await act(() => confirm.click());
    await vi.waitFor(() =>
      expect(dependencies.startIntermediatesOperation).toHaveBeenCalledWith(
        { previewId: 'preview-safe' },
        expect.anything()
      )
    );
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
    expect(host.textContent).toContain('intermediates.operation.status.running');
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.status.completed'), {
      timeout: 5_000,
    });
    expect(host.textContent).not.toContain('intermediates.operation.runAgain');
  });

  it('gates a force delete behind the disclosure, an acknowledgement and the typed word', async () => {
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    const dialog = await openDialog();
    expect(dialog.textContent).toContain('intermediates.dialog.keptReferenced');

    const toggles = () => [...dialog.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')];
    await act(() => toggles()[0]!.click());
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenLastCalledWith(
        expect.objectContaining({ mode: 'force' }),
        expect.anything()
      )
    );
    await vi.waitFor(() => expect(dialog.textContent).toContain('Portraits'));
    // Force mode keeps nothing for being referenced, so that reason is not listed at zero.
    expect(dialog.textContent).not.toContain('intermediates.dialog.keptReferenced');
    expect(dialog.textContent).toContain('intermediates.dialog.keptActive');
    const confirm = () => buttonWithText('intermediates.dialog.forceConfirm', dialog);
    expect(confirm().disabled).toBe(true);

    await act(() => toggles()[1]!.click());
    expect(confirm().disabled).toBe(true);

    const typed = dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])');
    expect(typed).not.toBeNull();
    await act(() => setInputValue(typed!, 'DELETE'));
    expect(confirm().disabled).toBe(false);
  });

  it('reports a failed load with a retry and an empty library plainly', async () => {
    dependencies.getIntermediatesSummary
      .mockReset()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValue(summaryOf([]));
    await renderManager();

    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.errors.couldNotLoad'));
    await act(() => buttonWithText('common.retry', host).click());
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.empty.title'));
  });

  it('preselects the requested project in single-user mode, where the session names no account', async () => {
    await renderManager({ currentUserId: null, focusProjectId: 'p2' });
    await vi.waitFor(() => expect(host.textContent).toContain('Landscapes'));

    expect(checkbox('intermediates.list.selectRow(name=Landscapes)').checked).toBe(true);
    expect(checkbox('intermediates.list.selectRow(name=Portraits)').checked).toBe(false);
    expect(isDeleteUnavailable()).toBe(false);
    expect(host.textContent).toContain('count=1');
  });

  it('keeps picks from another page in the estimate and the targets', async () => {
    const manyRows = Array.from({ length: 60 }, (_, index) => row(`p${index}`, `Project ${index}`, 1));
    dependencies.getIntermediatesSummary.mockReset().mockImplementation((params: { offset?: number }) => {
      const offset = params.offset ?? 0;
      const summary = summaryOf(manyRows.slice(offset, offset + 50));
      return Promise.resolve({ ...summary, offset, total: manyRows.length });
    });
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('Project 0'));

    await act(() => checkbox('intermediates.list.selectRow(name=Project 1)').click());
    await act(() => buttonWithText('common.nextPage', host).click());
    await vi.waitFor(() => expect(host.textContent).toContain('Project 55'));
    await act(() => checkbox('intermediates.list.selectRow(name=Project 55)').click());
    expect(host.textContent).toContain('count=2');

    await act(() => buttonWithText('intermediates.list.delete', host).click());
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenCalledWith(
        {
          mode: 'safe',
          scope: {
            kind: 'selection',
            targets: [
              { projectId: 'p1', userId: 'alice' },
              { projectId: 'p55', userId: 'alice' },
            ],
          },
        },
        expect.anything()
      )
    );
  });

  it('keeps all 119 other projects selected across pages and sends the exclusion, not the rows', async () => {
    const manyRows = Array.from({ length: 120 }, (_, index) => row(`p${index}`, `Project ${index}`, 1));
    const totals = summaryOf(manyRows).totals;
    dependencies.getIntermediatesSummary
      .mockReset()
      .mockImplementation((params: { offset?: number; limit?: number }) => {
        const offset = params.offset ?? 0;
        return Promise.resolve({
          ...summaryOf(manyRows.slice(offset, offset + (params.limit ?? 50))),
          offset,
          total: 120,
          totals,
        });
      });
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('Project 0'));
    await act(() => checkbox('intermediates.list.selectAll').click());
    await act(() => checkbox('intermediates.list.selectRow(name=Project 1)').click());
    await vi.waitFor(() => expect(host.textContent).toContain('count=119'));
    const nextPage = buttonWithText('common.nextPage', host);
    await act(() => {
      nextPage.focus();
      nextPage.click();
    });
    await vi.waitFor(() => expect(host.textContent).toContain('Project 55'));
    // The page flipped under the pager, which keeps keyboard focus for the next press.
    expect(document.activeElement).toBe(nextPage);
    expect(checkbox('intermediates.list.selectRow(name=Project 55)').checked).toBe(true);
    expect(checkbox('intermediates.list.selectAll').checked).toBe(false);
    await act(() => buttonWithText('common.nextPage', host).click());
    await vi.waitFor(() => expect(host.textContent).toContain('Project 100'));
    // The list is virtualized; the last row of the page mounts once it scrolls into view.
    await act(() => {
      const viewport = host.querySelector<HTMLElement>('[data-list-viewport]')!;
      viewport.scrollTop = viewport.scrollHeight;
      viewport.dispatchEvent(new Event('scroll'));
    });
    await vi.waitFor(() => expect(host.textContent).toContain('Project 119'));
    expect(checkbox('intermediates.list.selectRow(name=Project 119)').checked).toBe(true);
    await act(() => buttonWithText('intermediates.list.delete', host).click());
    const scope = matchingScope({ excluded: [{ projectId: 'p1', userId: 'alice' }] });
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenLastCalledWith(
        { mode: 'safe', scope },
        expect.anything()
      )
    );
    await act(() => document.querySelector<HTMLLabelElement>('[role="alertdialog"] label')!.click());
    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenLastCalledWith(
        { mode: 'force', scope },
        expect.anything()
      )
    );
    expect(dependencies.getIntermediatesSummary.mock.calls.every(([params]) => (params.limit ?? 50) <= 50)).toBe(true);
  });

  it('stops polling an operation the server no longer knows and lets it be dismissed', async () => {
    const { followIntermediatesOperation, activeOperationStore } =
      await import('@features/intermediates/data/operationStore');
    followIntermediatesOperation('missing');
    dependencies.getIntermediatesOperation.mockRejectedValue(new ApiError('Operation not found', 404));
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.lookupGone'));
    const calls = dependencies.getIntermediatesOperation.mock.calls.length;
    await new Promise((resolve) => {
      setTimeout(resolve, 2200);
    });
    expect(dependencies.getIntermediatesOperation).toHaveBeenCalledTimes(calls);
    await act(() => buttonWithText('intermediates.operation.dismiss', host).click());
    expect(activeOperationStore.getSnapshot().operationId).toBeNull();
    expect(document.activeElement).toBe(host.querySelector('[aria-label="intermediates.searchLabel"]'));
  });

  it('sends Select all under a search as a matching scope with its exclusions and no full read', async () => {
    const manyRows = Array.from({ length: 60 }, (_, index) => row(`p${index}`, `Match ${index}`, 1));
    dependencies.getIntermediatesSummary
      .mockReset()
      .mockImplementation((params: { offset?: number; limit?: number; search?: string }) => {
        const matching = params.search?.trim() ? manyRows : [...manyRows, row('other', 'Other', 1)];
        const offset = params.offset ?? 0;
        return Promise.resolve({
          ...summaryOf(matching),
          items: matching.slice(offset, offset + (params.limit ?? 50)),
          offset,
        });
      });
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('Match 0'));
    await act(() =>
      setInputValue(
        page.getByRole('textbox', { name: 'intermediates.searchLabel' }).element() as HTMLInputElement,
        'Match'
      )
    );
    await vi.waitFor(() => expect(host.textContent).not.toContain('Other'));
    await vi.waitFor(() => expect(host.querySelector('[role="list"]')!.getAttribute('aria-busy')).toBeNull());
    await act(() => checkbox('intermediates.list.selectAll').click());
    await act(() => checkbox('intermediates.list.selectRow(name=Match 3)').click());
    await vi.waitFor(() => expect(host.textContent).toContain('count=59'));
    await act(() => buttonWithText('intermediates.list.delete', host).click());

    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenCalledWith(
        { mode: 'safe', scope: matchingScope({ excluded: [{ projectId: 'p3', userId: 'alice' }], search: 'Match' }) },
        expect.anything()
      )
    );
    expect(dependencies.getIntermediatesSummary.mock.calls.every(([params]) => (params.limit ?? 50) <= 50)).toBe(true);
  });

  it('opens the confirmation on Cancel rather than the force toggle', async () => {
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('Portraits'));
    await act(() => buttonWithText('intermediates.list.delete', host).click());
    const dialog = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>('[role="alertdialog"]');
      expect(element).not.toBeNull();
      return element!;
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(buttonWithText('common.cancel', dialog)));
    const description = document.getElementById(dialog.getAttribute('aria-describedby')!)!;
    expect(description.getAttribute('aria-live')).toBe('polite');
    await vi.waitFor(() => expect(description.textContent).toContain('intermediates.dialog.reclaim'));
    expect(description.getAttribute('aria-busy')).toBeNull();
  });

  it('holds Delete, focusable with its reason, until the followed cleanup settles', async () => {
    const { followIntermediatesOperation } = await import('@features/intermediates/data/operationStore');
    followIntermediatesOperation('op-1');
    const publish = await followOperation(operationOf('running'));
    await renderManager({ focusProjectId: 'p1' });
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.status.running'));
    const deleteButton = buttonWithText('intermediates.list.delete', host);
    expect(isDeleteUnavailable()).toBe(true);
    deleteButton.focus();
    expect(document.activeElement).toBe(deleteButton);
    const reason = document.getElementById(deleteButton.getAttribute('aria-describedby')!);
    expect(reason?.textContent).toBe('intermediates.list.deleteWaiting');
    await act(() => deleteButton.click());
    expect(dependencies.createIntermediatesPreview).not.toHaveBeenCalled();

    await publish(operationOf('completed'));
    expect(host.textContent).toContain('intermediates.operation.status.completed');
    expect(isDeleteUnavailable()).toBe(false);
    expect(deleteButton.hasAttribute('aria-describedby')).toBe(false);
    expect(host.textContent).not.toContain('intermediates.list.deleteWaiting');
  });

  it('offers to delete again after a run that left failures, in the same mode and scope', async () => {
    const { followIntermediatesOperation } = await import('@features/intermediates/data/operationStore');
    followIntermediatesOperation('op-1');
    const failed: IntermediatesOperation = {
      ...operationOf('completed'),
      mode: 'force',
      progress: { ...operationOf('completed').progress, failedImages: 2 },
      scope: matchingScope({ excluded: [{ projectId: 'p2', userId: 'alice' }], search: 'Port' }),
    };
    await followOperation(failed);
    await renderManager();
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.status.completed'));
    const runAgain = buttonWithText('intermediates.operation.runAgain', host);

    await act(() => runAgain.click());

    await vi.waitFor(() =>
      expect(dependencies.createIntermediatesPreview).toHaveBeenCalledWith(
        { mode: 'force', scope: failed.scope },
        expect.anything()
      )
    );
    const dialog = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>('[role="alertdialog"]');
      expect(element).not.toBeNull();
      return element!;
    });
    // A force run repeats as a force preview, behind the same acknowledgement and typed word.
    await vi.waitFor(() => expect(dialog.textContent).toContain('intermediates.dialog.affectedClientState'));
    expect(buttonWithText('intermediates.dialog.forceConfirm', dialog).disabled).toBe(true);
    await act(() => buttonWithText('common.cancel', dialog).click());
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).toBeNull());
    await vi.waitFor(() => expect(document.activeElement?.textContent).toBe('intermediates.operation.runAgain'), {
      timeout: 3_000,
    });
    expect(host.contains(document.activeElement)).toBe(true);
  });

  it('starts over on a new entry point request while its settings section is already showing', async () => {
    host = document.createElement('div');
    host.style.height = '640px';
    host.style.display = 'flex';
    host.style.flexDirection = 'column';
    document.body.append(host);
    root = createRoot(host);
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { requestIntermediatesFocus } = await import('@features/intermediates/data/focus');
    const { IntermediatesSettingsField } = await import('./IntermediatesSettingsField');
    requestIntermediatesFocus({ projectId: 'p1' });
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <IntermediatesSettingsField field={INTERMEDIATES_FIELD} surface="dialog" />
          </QueryClientProvider>
        </ChakraProvider>
      );
    });
    await vi.waitFor(() => expect(checkbox('intermediates.list.selectRow(name=Portraits)').checked).toBe(true));
    expect(checkbox('intermediates.list.selectRow(name=Landscapes)').checked).toBe(false);

    // The Launchpad keeps the Preferences page mounted, so a later request arrives while the manager is showing.
    await act(() => requestIntermediatesFocus({ projectId: 'p2' }));
    await vi.waitFor(() => expect(checkbox('intermediates.list.selectRow(name=Landscapes)').checked).toBe(true));
    expect(checkbox('intermediates.list.selectRow(name=Portraits)').checked).toBe(false);
  });

  it('asks the server once under StrictMode, follows the running operation and consumes the entry point intent on commit', async () => {
    const { peekIntermediatesFocus } = await import('@features/intermediates/data/focus');
    const running = { ...operationOf('running'), operationId: 'op-running' };
    dependencies.listIntermediatesOperations.mockResolvedValue([running]);
    dependencies.getIntermediatesOperation.mockResolvedValue(running);

    await renderManager({ focusProjectId: 'p2', strict: true });
    await vi.waitFor(() => expect(host.textContent).toContain('intermediates.operation.status.running'));
    expect(dependencies.listIntermediatesOperations).toHaveBeenCalledOnce();
    expect(checkbox('intermediates.list.selectRow(name=Landscapes)').checked).toBe(true);
    expect(peekIntermediatesFocus()).toBeNull();
  });
});
