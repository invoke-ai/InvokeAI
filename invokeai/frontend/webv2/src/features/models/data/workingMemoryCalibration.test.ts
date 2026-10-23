import type { ConnectionListener } from '@platform/transport/socketHub';

import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const hub = vi.hoisted(() => ({ statusChanged: null as ((payload: unknown) => void) | null }));

vi.mock('@platform/transport/socketHub', () => ({
  socketHub: {
    on: (event: string, listener: (payload: unknown) => void) => {
      if (event === 'queue_item_status_changed') {
        hub.statusChanged = listener;
      }
      return () => {
        hub.statusChanged = null;
      };
    },
    onConnectionChange: (listener: ConnectionListener) => {
      listener('connected');
      return () => undefined;
    },
  },
}));

const { attachWorkingMemoryRealtime } = await import('./workingMemoryCalibration');
const { workingMemoryKeys } = await import('./workingMemoryQueries');
const { calibrationRunOrigin } = await import('@features/models/core/workingMemory');
const { accountLifecycle, captureAccountScope } = await import('@platform/state/accountLifecycle');

let queryClient: QueryClient;
let detach: () => void;

beforeEach(() => {
  accountLifecycle.activate('wm-realtime-test', ':user:wm-realtime-test');
  queryClient = new QueryClient();
  detach = attachWorkingMemoryRealtime(queryClient);
});

afterEach(() => {
  detach();
  queryClient.clear();
  accountLifecycle.invalidate();
});

it("marks every model's plan stale when a calibration item changes status, without reading one then", () => {
  const owner = captureAccountScope();
  const plans = [workingMemoryKeys.plan(owner, 'sdxl'), workingMemoryKeys.plan(owner, 'flux')];
  const read = vi.fn(() => Promise.resolve({}));
  const unsubscribe = plans.map((key) => {
    queryClient.setQueryData(key, { supported: true });
    // An observed plan, as an open section has: an invalidation that refetched would read it now.
    return new QueryObserver(queryClient, { queryFn: read, queryKey: key, staleTime: Infinity }).subscribe(
      () => undefined
    );
  });

  hub.statusChanged?.({ origin: 'generation' });
  expect(plans.map((key) => queryClient.getQueryState(key)?.isInvalidated)).toEqual([false, false]);

  hub.statusChanged?.({ origin: calibrationRunOrigin('run-1'), status: 'completed' });

  expect(plans.map((key) => queryClient.getQueryState(key)?.isInvalidated)).toEqual([true, true]);
  expect(read).not.toHaveBeenCalled();
  for (const stop of unsubscribe) {
    stop();
  }
});
