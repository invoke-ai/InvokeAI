import type { ModelConfig } from '@features/models/core/types';

import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';

import { getFp8StorageSupport, type Fp8StorageSupportRow } from './api';

/**
 * Which models may be offered the FP8 Storage default, answered by the backend per `(base, type, format)`. Neither type
 * nor base is enough: a GGUF main model's packed weights must never be re-encoded, and FLUX main casts while a FLUX
 * ControlNet's loader never does.
 *
 * Static per backend build, so it is fetched once on first use rather than kept fresh.
 */

export interface Fp8StorageSupportSnapshot {
  /** Null until fetched. Keyed `base\u0000type\u0000format`. */
  byKey: ReadonlyMap<string, boolean> | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
}

const EMPTY_SNAPSHOT: Fp8StorageSupportSnapshot = { byKey: null, status: 'idle' };
const store = createExternalStore<Fp8StorageSupportSnapshot>(EMPTY_SNAPSHOT);
const refreshFlight = createTrailingSingleFlight();

const rowKey = (base: string, type: string, format: string): string => `${base}\u0000${type}\u0000${format}`;

registerAccountOwnedResource({
  clear: () => {
    refreshFlight.reset();
    store.setSnapshot(EMPTY_SNAPSHOT);
  },
  name: 'fp8-storage-support',
});

const refresh = (): Promise<void> =>
  refreshFlight.run(() => {
    const owner = captureAccountScope();

    store.patchSnapshot({ status: store.getSnapshot().byKey ? 'loaded' : 'loading' });

    return getFp8StorageSupport(owner.signal)
      .then((rows: Fp8StorageSupportRow[]) => {
        if (isAccountScopeCurrent(owner)) {
          store.setSnapshot({
            byKey: new Map(rows.map((row) => [rowKey(row.base, row.type, row.format), row.supported])),
            status: 'loaded',
          });
        }
      })
      .catch(() => {
        if (isAccountScopeCurrent(owner)) {
          // No error copy: the only consumer hides a control it cannot vouch for, which needs no message. Recorded as
          // `error` so a later mount retries rather than waiting forever.
          store.patchSnapshot({ status: store.getSnapshot().byKey ? 'loaded' : 'error' });
        }
      });
  });

/** Fetch on first use; concurrent callers share the request. */
export const ensureFp8StorageSupportLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refresh();
  }

  return refreshFlight.inflight() ?? Promise.resolve();
};

/**
 * Whether FP8 Storage does anything for this model. `false` until the table has arrived.
 *
 * Unknown reads as unsupported in both directions — before the table loads, and for a key the table does not contain:
 * an inert control is a defect, a control that appears a moment late is not.
 *
 * A row with base `any` is a wildcard loader registration, resolved after the exact key and in the same order the
 * backend registry resolves a loader. T2I adapters are served entirely that way, so without the fallback they would all
 * lose the control.
 */
export const isFp8StorageSupported = (
  snapshot: Fp8StorageSupportSnapshot,
  model: Pick<ModelConfig, 'base' | 'format' | 'type'>
): boolean =>
  snapshot.byKey?.get(rowKey(model.base, model.type, model.format)) ??
  snapshot.byKey?.get(rowKey('any', model.type, model.format)) ??
  false;

export const getFp8StorageSupportSnapshot = (): Fp8StorageSupportSnapshot => store.getSnapshot();

export const useFp8StorageSupportSelector = store.useSelector;
