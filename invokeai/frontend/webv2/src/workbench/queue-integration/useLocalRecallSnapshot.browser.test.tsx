import { buildQueueItemOrigin } from '@features/queue/contracts';
import { createDefaultVideoWidgetValues } from '@features/video';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

import { createAccountOwnedQueueRecallCache } from './queueRecallCache';
import { useLocalRecallSnapshot } from './useLocalRecallSnapshot';

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchSelector: (selector: (state: { projects: [] }) => unknown) => selector({ projects: [] }),
}));

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

it('restores exact video recall after reload without loading an open project', async () => {
  accountLifecycle.activate(`recall-hook-${crypto.randomUUID()}`);
  const cache = await createAccountOwnedQueueRecallCache(captureAccountScope());
  expect(
    await cache.put({
      projectId: 'p',
      queueItemId: 'q',
      sourceId: 'video',
      submittedAt: '2026-09-04T00:00:00Z',
      videoValues: { ...createDefaultVideoWidgetValues(), positivePrompt: 'durable video prompt' },
    })
  ).toEqual({ kind: 'stored' });
  cache.close();
  const Probe = () => {
    const recall = useLocalRecallSnapshot(buildQueueItemOrigin('q', 'p'));
    return <span>{recall?.videoValues?.positivePrompt ?? 'missing'}</span>;
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  await act(() =>
    root!.render(
      <QueryClientProvider client={client!}>
        <Probe />
      </QueryClientProvider>
    )
  );
  await vi.waitFor(() => expect(host?.textContent).toBe('durable video prompt'));
});
