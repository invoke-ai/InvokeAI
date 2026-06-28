import { getApiErrorMessage } from '@workbench/backend/http';
import { createExternalStore } from '@workbench/externalStore';
import { shallowEqual } from '@workbench/workbenchSelectors';

import type { QueueAndProcessorStatus, QueueQueryScope, QueueServerItem, QueueStatusCounts } from './queueServerApi';

import { getCurrentBatchItems } from './currentBatchItems';
import { clearProgressForQueueItemsWithResults } from './queueProgressCleanup';
import {
  getCurrentQueueItem,
  getNextQueueItem,
  getQueueItemIds,
  getQueueItemsByIds,
  getQueueStatus,
} from './queueServerApi';

/**
 * Read model for the whole backend session queue, shared by the Queue widget's
 * header, stats, CURRENT BATCH, and RECENT sections. A single external store fed by
 * one debounced `refreshQueue()` keeps every section consistent and lets each
 * subscribe to just the slice it renders (composition: one lifted store, many
 * thin readers). Socket events drive refresh via `QueueDataRuntime`.
 */

/** How many of the most-recent items we hydrate for the RECENT list. */
export const RECENT_WINDOW = 50;
const REFRESH_DEBOUNCE_MS = 300;

const EMPTY_COUNTS: QueueStatusCounts = {
  canceled: 0,
  completed: 0,
  failed: 0,
  in_progress: 0,
  pending: 0,
  queue_id: 'default',
  total: 0,
};

interface QueueDataSnapshot {
  status: QueueAndProcessorStatus | null;
  current: QueueServerItem | null;
  next: QueueServerItem | null;
  items: QueueServerItem[];
  scope: QueueQueryScope;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const store = createExternalStore<QueueDataSnapshot>({
  current: null,
  error: null,
  items: [],
  loadState: 'idle',
  next: null,
  scope: {},
  status: null,
});

let inflight: Promise<void> | null = null;
let refreshTimer: ReturnType<typeof setTimeout> | null = null;
let needsRefresh = false;
let refreshSequence = 0;

const runRefresh = async (): Promise<void> => {
  const scope = store.getSnapshot().scope;
  const sequence = ++refreshSequence;

  if (store.getSnapshot().loadState === 'idle') {
    store.patchSnapshot({ loadState: 'loading' });
  }

  try {
    // Independent reads run together to avoid a request waterfall; the window
    // fetch is the only one that must wait (it needs the ordered ids first).
    const [status, current, next, idsResult] = await Promise.all([
      getQueueStatus(scope),
      getCurrentQueueItem(scope),
      getNextQueueItem(scope),
      getQueueItemIds('desc', scope),
    ]);
    const items = await getQueueItemsByIds(idsResult.item_ids.slice(0, RECENT_WINDOW));

    if (sequence === refreshSequence) {
      clearProgressForQueueItemsWithResults([current, next, ...items].filter((item) => item !== null));
      store.patchSnapshot({ current, error: null, items, loadState: 'loaded', next, status });
    }
  } catch (error) {
    if (sequence === refreshSequence) {
      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load queue'), loadState: 'error' });
    }
  }
};

const runRefreshLoop = async (): Promise<void> => {
  do {
    needsRefresh = false;
    await runRefresh();
  } while (needsRefresh);
};

/** Coalesced refresh — many socket events in a burst collapse into one fetch. */
export const refreshQueue = (): Promise<void> => {
  if (inflight !== null && refreshTimer === null) {
    needsRefresh = true;
    return inflight;
  }

  if (refreshTimer !== null) {
    return inflight ?? Promise.resolve();
  }

  inflight = new Promise<void>((resolve) => {
    refreshTimer = setTimeout(() => {
      refreshTimer = null;
      void runRefreshLoop().finally(() => {
        inflight = null;
        resolve();
      });
    }, REFRESH_DEBOUNCE_MS);
  });

  return inflight;
};

export const setQueueScope = (scope: QueueQueryScope): void => {
  const previous = store.getSnapshot().scope;
  const isSameScope = previous.originPrefix === scope.originPrefix;

  if (isSameScope) {
    return;
  }

  store.patchSnapshot({ current: null, error: null, items: [], loadState: 'loading', next: null, scope, status: null });
  void refreshQueue();
};

/** First load — idempotent; safe to call from every mounting surface. */
export const ensureQueueLoaded = (): void => {
  if (store.getSnapshot().loadState === 'idle') {
    void refreshQueue();
  }
};

// --- Selectors / hooks -----------------------------------------------------

export const useQueueCounts = (): QueueStatusCounts =>
  store.useSelector((snapshot) => snapshot.status?.queue ?? EMPTY_COUNTS, shallowEqual);

/** The processor is "paused" when it has been stopped (`is_started` false). */
export const useIsProcessorPaused = (): boolean =>
  store.useSelector((snapshot) => snapshot.status?.processor.is_started === false);

export const useNowNextItems = (): { current: QueueServerItem | null; next: QueueServerItem | null } =>
  store.useSelector((snapshot) => ({ current: snapshot.current, next: snapshot.next }), shallowEqual);

export const useCurrentBatchItems = (): QueueServerItem[] =>
  store.useSelector(
    (snapshot) => getCurrentBatchItems({ current: snapshot.current, items: snapshot.items, next: snapshot.next }),
    shallowEqual
  );

export const useRecentItems = (): QueueServerItem[] => store.useSelector((snapshot) => snapshot.items);

export const useQueueLoadState = (): { loadState: QueueDataSnapshot['loadState']; error: string | null } =>
  store.useSelector((snapshot) => ({ error: snapshot.error, loadState: snapshot.loadState }), shallowEqual);
