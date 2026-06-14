import { useSyncExternalStore } from 'react';

/**
 * The most recent denoising preview image from `invocation_progress` events,
 * as a b64 data URL. Cleared when the run settles so consumers (the editor's
 * Current Image node, progress surfaces) fall back to the last real output.
 */

export interface ProgressImageSnapshot {
  dataUrl: string;
  width: number;
  height: number;
}

let snapshot: ProgressImageSnapshot | null = null;
const listeners = new Set<() => void>();

const emit = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

export const progressImageStore = {
  clear(): void {
    if (snapshot !== null) {
      snapshot = null;
      emit();
    }
  },
  set(image: ProgressImageSnapshot): void {
    snapshot = image;
    emit();
  },
};

export type ProgressImageSink = typeof progressImageStore;

export const useProgressImage = (): ProgressImageSnapshot | null => useSyncExternalStore(subscribe, () => snapshot);
