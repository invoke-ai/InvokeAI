import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson } from '@platform/transport/http';

import { getRemoteWorkersSettings, isRemoteWorkerEnabled } from './remoteWorkersStore';

export type RemoteWorkerStatus = 'checking' | 'online' | 'offline' | 'login_required';
type RemoteWorkerHealth = { status: RemoteWorkerStatus; checkedAt: number };

const HEALTH_STALE_MS = 15_000;
export const remoteWorkersHealthStore = createExternalStore<{ byUrl: Record<string, RemoteWorkerHealth> }>({
  byUrl: {},
});
const inFlight = new Map<string, Promise<void>>();
const healthRevisionByUrl = new Map<string, number>();

registerAccountOwnedResource({
  name: 'remote-workers-health',
  clear: () => {
    inFlight.clear();
    healthRevisionByUrl.clear();
    remoteWorkersHealthStore.setSnapshot({ byUrl: {} });
  },
});

/**
 * Discards cached/in-flight health for one worker after its authentication changes.
 * Any older probe may still finish at the transport layer, but its result cannot
 * overwrite the next probe because the per-URL revision has changed.
 */
export const invalidateRemoteWorkerHealth = (url: string): void => {
  healthRevisionByUrl.set(url, (healthRevisionByUrl.get(url) ?? 0) + 1);
  inFlight.delete(url);

  const { byUrl } = remoteWorkersHealthStore.getSnapshot();
  if (!(url in byUrl)) {
    return;
  }
  const next = { ...byUrl };
  delete next[url];
  remoteWorkersHealthStore.setSnapshot({ byUrl: next });
};

/** Probes the primary's authenticated status endpoint; the browser never fetches a worker URL directly. */
export const refreshRemoteWorkerHealth = async (urls: readonly string[]): Promise<void> => {
  const owner = captureAccountScope();
  if (!owner.accountId || !getRemoteWorkersSettings().enabled) {
    return;
  }
  await Promise.all(
    [...new Set(urls)].map(async (url) => {
      // Settings can change between starting the batch and reaching this URL.
      if (!getRemoteWorkersSettings().enabled || !isRemoteWorkerEnabled(url)) {
        return;
      }
      const running = inFlight.get(url);
      if (running) {
        await running;
        return;
      }
      const prior = remoteWorkersHealthStore.getSnapshot().byUrl[url];
      if (prior && prior.status !== 'checking' && Date.now() - prior.checkedAt < HEALTH_STALE_MS) {
        return;
      }
      if (!prior) {
        remoteWorkersHealthStore.setSnapshot({
          byUrl: { ...remoteWorkersHealthStore.getSnapshot().byUrl, [url]: { status: 'checking', checkedAt: 0 } },
        });
      }
      const revision = healthRevisionByUrl.get(url) ?? 0;
      const task = (async () => {
        let status: RemoteWorkerStatus;
        try {
          if (!getRemoteWorkersSettings().enabled || !isRemoteWorkerEnabled(url)) {
            return;
          }
          const result = await apiFetchJson<{ status: RemoteWorkerStatus }>(
            `/api/v1/remote_workers/status?url=${encodeURIComponent(url)}`,
            { signal: owner.signal }
          );
          status = result.status === 'online' || result.status === 'login_required' ? result.status : 'offline';
        } catch {
          status = 'offline';
        }
        // An already-running request may finish after the user switches the feature off.
        // Never mark such a worker online from a result received while disabled.
        if (
          isAccountScopeCurrent(owner) &&
          getRemoteWorkersSettings().enabled &&
          isRemoteWorkerEnabled(url) &&
          (healthRevisionByUrl.get(url) ?? 0) === revision
        ) {
          remoteWorkersHealthStore.setSnapshot({
            byUrl: { ...remoteWorkersHealthStore.getSnapshot().byUrl, [url]: { status, checkedAt: Date.now() } },
          });
        }
      })();
      inFlight.set(url, task);
      try {
        await task;
      } finally {
        if (inFlight.get(url) === task) {
          inFlight.delete(url);
        }
      }
    })
  );
};
