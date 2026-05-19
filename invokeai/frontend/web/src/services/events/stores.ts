import { round } from 'es-toolkit/compat';
import { atom, computed } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket, LLMTaskProgressEventPayload } from 'services/events/types';

export const $socket = atom<AppSocket | null>(null);
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $loadingModelsCount = atom<number>(0);

// LLM utility task progress (expand-prompt, image-to-prompt). Keyed by task_id so
// concurrent tasks don't clobber each other. Components subscribe and read the
// entry matching their current task.
type LLMTaskState =
  | { status: 'progress'; payload: LLMTaskProgressEventPayload }
  | { status: 'complete' }
  | { status: 'error'; error: string };

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
