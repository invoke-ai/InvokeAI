import { createExternalStore } from '@workbench/externalStore';

/**
 * A short "reveal window" after a generation completes: consumers that show
 * live denoise frames suppress them while the hold is armed, so the finished
 * result is actually seen before the next queue item's noise takes over. Armed
 * by the queue coordinator when a backend item completes while more items are
 * still running; expires on its own timer (the timer lives here, not in a
 * component effect).
 */

export const REVEAL_HOLD_DURATION_MS = 2000;

const store = createExternalStore<{ holdUntil: number }>({ holdUntil: 0 });

let timer: ReturnType<typeof setTimeout> | null = null;

export const revealHoldStore = {
  arm(durationMs: number = REVEAL_HOLD_DURATION_MS): void {
    store.patchSnapshot({ holdUntil: Date.now() + durationMs });

    if (timer) {
      clearTimeout(timer);
    }

    timer = setTimeout(() => {
      timer = null;
      store.patchSnapshot({ holdUntil: 0 });
    }, durationMs);
  },
  clear(): void {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }

    store.patchSnapshot({ holdUntil: 0 });
  },
};

export type RevealHoldSink = Pick<typeof revealHoldStore, 'arm' | 'clear'>;

export const isRevealHolding = (): boolean => store.getSnapshot().holdUntil > Date.now();

/** Whether a reveal hold is currently active. Re-evaluates when the hold arms or expires. */
export const useRevealHold = (): boolean => store.useSelector((snapshot) => snapshot.holdUntil > Date.now());
