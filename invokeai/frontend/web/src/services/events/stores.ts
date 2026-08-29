import { round } from 'es-toolkit/compat';
import { atom, computed, map } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket, LLMTaskProgressEventPayload } from 'services/events/types';

export const $socket = atom<AppSocket | null>(null);
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $loadingModelsCount = atom<number>(0);

// LLM utility task progress (expand-prompt, image-to-prompt). Keyed by task_id so
// concurrent tasks don't clobber each other. Components subscribe and read the
// entry matching their current task. Only the in-flight progress state is tracked:
// completion and error clear the entry (errors surface to the user via the RTK Query
// toast), so a late socket event can never orphan an entry in the store.
type LLMTaskState = { status: 'progress'; payload: LLMTaskProgressEventPayload };

export const $llmTaskStates = atom<Record<string, LLMTaskState>>({});

export const setLLMTaskState = (taskId: string, state: LLMTaskState): void => {
  $llmTaskStates.set({ ...$llmTaskStates.get(), [taskId]: state });
};

export const clearLLMTaskState = (taskId: string): void => {
  const current = $llmTaskStates.get();
  if (!(taskId in current)) {
    return;
  }
  const next = { ...current };
  delete next[taskId];
  $llmTaskStates.set(next);
};

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
