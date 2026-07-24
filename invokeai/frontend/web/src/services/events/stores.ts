import { round } from 'es-toolkit/compat';
import { atom, computed, map } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';

export const $socket = atom<AppSocket | null>(null);
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $loadingModelsCount = atom<number>(0);

/**
 * Live progress events keyed by queue item id. Unlike `$lastProgressEvent` (a single global value that
 * is overwritten by whichever session reported last), this tracks each in-flight session separately so
 * the UI can render one progress bar per concurrent session (multi-GPU). Entries are added as progress
 * events arrive and removed when the session reaches a terminal state.
 */
const $progressEvents = map<Record<number, S['InvocationProgressEvent'] | undefined>>({});

/** In-flight sessions sorted by queue item id, for a stable top-to-bottom bar order. */
export const $activeProgressEvents = computed($progressEvents, (events) =>
  Object.values(events)
    .filter((event): event is S['InvocationProgressEvent'] => event !== undefined)
    .sort((a, b) => a.item_id - b.item_id)
);

export const setProgressEvent = (event: S['InvocationProgressEvent']) => {
  $progressEvents.setKey(event.item_id, event);
};

export const clearProgressEvent = (itemId: number) => {
  $progressEvents.setKey(itemId, undefined);
};

export const clearAllProgressEvents = () => {
  $progressEvents.set({});
};

export const $lastProgressMessage = computed($lastProgressEvent, (val) => {
  if (!val) {
    return null;
  }
  return formatProgressMessage(val);
});
export const formatProgressMessage = (data: S['InvocationProgressEvent']): string => {
  let message = data.message;
  if (data.percentage) {
    message += ` (${round(data.percentage * 100)}%)`;
  }
  return message;
};
