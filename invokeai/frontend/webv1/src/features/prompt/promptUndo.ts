import { atom } from 'nanostores';

type PromptUndoState = {
  previousPrompt: string;
  timestamp: number;
};

const UNDO_TIMEOUT_MS = 30_000;

const $promptUndo = atom<PromptUndoState | null>(null);

let timeoutId: ReturnType<typeof setTimeout> | null = null;

/**
 * Save the current prompt so it can be restored with Ctrl+Z.
 * Automatically expires after 30 seconds.
 */
export const setPromptUndo = (previousPrompt: string): void => {
  if (timeoutId) {
    clearTimeout(timeoutId);
  }
  $promptUndo.set({ previousPrompt, timestamp: Date.now() });
  timeoutId = setTimeout(() => {
    $promptUndo.set(null);
    timeoutId = null;
  }, UNDO_TIMEOUT_MS);
};

/**
 * Clear the undo state (e.g., when the user types manually).
 */
export const clearPromptUndo = (): void => {
  if (timeoutId) {
    clearTimeout(timeoutId);
    timeoutId = null;
  }
  $promptUndo.set(null);
};

/**
 * Consume the undo state: returns the previous prompt and clears it.
 * Returns null if no undo is available or it has expired.
 */
export const consumePromptUndo = (): string | null => {
  const state = $promptUndo.get();
  if (!state) {
    return null;
  }
  if (Date.now() - state.timestamp > UNDO_TIMEOUT_MS) {
    clearPromptUndo();
    return null;
  }
  const { previousPrompt } = state;
  clearPromptUndo();
  return previousPrompt;
};
