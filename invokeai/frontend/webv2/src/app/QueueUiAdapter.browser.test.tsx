import type * as QueuePublicApi from '@features/queue/publicApi';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';
import type { ComponentType } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { loadQueueWidgetImplementation } from '@features/queue/widget';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import { QueueUiAdapterProvider } from './QueueUiAdapter';

let store: ReturnType<typeof createWorkbenchStore>;

vi.mock('@features/queue/publicApi', async (importOriginal) => ({
  ...(await importOriginal<typeof QueuePublicApi>()),
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown, isEqual = shallowEqual) =>
    useExternalStoreSelector(
      store.subscribe,
      store.getSnapshot,
      (snapshot) => selector(snapshot.activeProject),
      isEqual
    ),
  useWorkbenchCommands: () => store.commands,
  useWorkbenchSelector: (selector: (snapshot: WorkbenchSnapshot) => unknown, isEqual = shallowEqual) =>
    useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const RUN_ID = 'local-run';
let projectId: string;
let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  accountLifecycle.activate('queue-ui-adapter');
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  projectId = project.id;
  // Queued long ago (restored from storage, say) and never accepted, so the widget lists it at once.
  project.queue = {
    items: [
      {
        cancellable: true,
        id: RUN_ID,
        snapshot: {
          presentation: { batchCount: 1, height: 512, positivePrompt: 'A lighthouse in fog', width: 512 },
          sourceId: 'generate',
          submittedAt: '2026-01-01T00:00:00.000Z',
        } as Project['queue']['items'][number]['snapshot'],
        status: 'pending',
      },
    ],
  };
  store = createWorkbenchStore(state);
  store.commands.queue.setConnectionStatus({ status: 'connected' });
  host = document.createElement('div');
  host.style.cssText = 'width:360px;height:600px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

const rowText = () =>
  host.querySelector('[role="list"][aria-label="Submitting"] > [role="listitem"]')?.textContent ?? null;

const recordServerError = (failureCount: number) =>
  act(() =>
    store.commands.queue.recordSubmissionRetry({
      projectId,
      queueItemId: RUN_ID,
      retry: {
        failedAt: Date.now(),
        failureCount,
        message: 'Internal Server Error',
        reason: 'server-error',
        serverErrorCount: failureCount,
        serverErrorSince: Date.now(),
        status: 500,
      },
    })
  );

it("lists the Workbench's unaccepted run, follows each recorded retry, and drops it once the backend accepts it", async () => {
  const { view: QueueWidgetView } = (await loadQueueWidgetImplementation()) as {
    view: ComponentType<{ region: 'right' }>;
  };
  const client = new QueryClient();
  client.setQueryData(['queue'], {
    current: null,
    items: [],
    next: null,
    scope: {},
    status: {
      processor: { isProcessing: false, isStarted: true },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: 0,
        pending: 0,
        queueId: 'default',
        total: 0,
        waiting: 0,
      },
    },
  });

  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <QueueUiAdapterProvider>
              <QueueWidgetView region="right" />
            </QueueUiAdapterProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

  await expect.poll(rowText).toContain('A lighthouse in fog');
  expect(rowText()).toContain('Sending to the server…');

  await recordServerError(1);
  expect(rowText()).toContain('Server error, retrying (attempt 2)');

  // Only the retry changed: the run must still re-render with it.
  await recordServerError(2);
  expect(rowText()).toContain('Server error, retrying (attempt 3)');

  await act(() => store.commands.queue.markBackendSubmitted({ backendItemIds: [42], projectId, queueItemId: RUN_ID }));
  expect(rowText()).toBeNull();
  expect(host.textContent).not.toContain('Submitting');
});
