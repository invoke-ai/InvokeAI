import { buildQueueItemOrigin } from '@features/queue/contracts';
import { createDefaultVideoWidgetValues } from '@features/video';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({
  close: vi.fn(),
  get: vi.fn(),
}));

vi.mock('./queueRecallCache', () => ({
  createAccountOwnedQueueRecallCache: vi.fn(() =>
    Promise.resolve({
      close: harness.close,
      get: harness.get,
    })
  ),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchSelector: (selector: (state: { projects: [] }) => unknown) => selector({ projects: [] }),
}));

import { useLocalRecallSnapshot } from './useLocalRecallSnapshot';

let root: Root | undefined;
let host: HTMLDivElement | undefined;
let client: QueryClient | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  client?.clear();
  accountLifecycle.invalidate();
});

it('keeps recall pending until the durable source resolves, then identifies video', async () => {
  accountLifecycle.activate(`recall-pending-${crypto.randomUUID()}`);
  let resolveGet: ((value: unknown) => void) | undefined;
  harness.get.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        resolveGet = resolve;
      })
  );
  const Probe = () => {
    const recall = useLocalRecallSnapshot(buildQueueItemOrigin('q', 'p'));
    return <span>{recall === undefined ? 'pending' : (recall?.sourceId ?? 'missing')}</span>;
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  client = new QueryClient({ defaultOptions: { queries: { gcTime: 0, retry: false } } });
  await act(() =>
    root!.render(
      <QueryClientProvider client={client!}>
        <Probe />
      </QueryClientProvider>
    )
  );

  expect(host.textContent).toBe('pending');
  await act(async () => {
    resolveGet?.({
      kind: 'found',
      snapshot: {
        projectId: 'p',
        queueItemId: 'q',
        sourceId: 'video',
        submittedAt: '2026-09-04T00:00:00Z',
        videoValues: createDefaultVideoWidgetValues(),
      },
    });
    await vi.waitFor(() => expect(host?.textContent).toBe('video'));
  });
});
