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
 * Which models may be offered the FP8 Storage default, answered by the backend.
 *
 * The client used to answer it from the model type alone — `main`, `controlnet` or `t2i_adapter` — and
 * so offered the control for 19 of the 52 loader keys that ignore it: a GGUF main model whose packed
 * weights must never be re-encoded, a FLUX ControlNet whose loader never casts. Type is not enough,
 * and neither is base: FLUX main casts and FLUX ControlNet does not. The server composes the whole
 * answer per `(base, type, format)` and ships the finished boolean.
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
          // No error copy: the only consumer hides a control it cannot vouch for, which needs no
          // message. Recorded as `error` so a later mount retries rather than waiting forever.
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
 * Unknown reads as unsupported in both directions — before the table loads, and for a key the table
 * does not contain. Offering a control that turns out to be inert is the defect this replaces; a
 * control that appears a moment late is not.
 *
 * A row with base `any` is a wildcard loader registration, resolved after the exact key and in the same
 * order the backend registry resolves a loader. T2I adapters are served entirely that way, so without
 * the fallback they would all lose the control.
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
