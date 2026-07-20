import type { CanvasDiagnosticsSnapshot } from './capabilities';

export type { CanvasDiagnosticsSnapshot } from './capabilities';

export type CanvasDiagnosticsCounter = keyof CanvasDiagnosticsSnapshot;
type MutableCanvasDiagnosticsSnapshot = { -readonly [K in CanvasDiagnosticsCounter]: number };

export interface CanvasDiagnostics {
  readonly enabled: boolean;
  increment(counter: CanvasDiagnosticsCounter): void;
  add(counter: CanvasDiagnosticsCounter, amount: number): void;
  snapshot(): Readonly<CanvasDiagnosticsSnapshot>;
  reset(): void;
}

const zeroSnapshot = (): MutableCanvasDiagnosticsSnapshot => ({
  allocatedBaseBytes: 0,
  allocatedDerivedBytes: 0,
  compositeFrames: 0,
  derivedCacheEvictions: 0,
  derivedCacheHits: 0,
  derivedCacheMisses: 0,
  imageDataReads: 0,
  imageDataWrites: 0,
  layersConsidered: 0,
  layersCulled: 0,
  layersDrawn: 0,
  overlayFrames: 0,
  overBudgetVisibleBaseBytes: 0,
  surfaceCreations: 0,
  surfaceResizes: 0,
});

const DISABLED_SNAPSHOT = Object.freeze(zeroSnapshot());

export const createCanvasDiagnostics = (enabled = false): CanvasDiagnostics => {
  if (!enabled) {
    return {
      add: () => undefined,
      enabled: false,
      increment: () => undefined,
      reset: () => undefined,
      snapshot: () => DISABLED_SNAPSHOT,
    };
  }

  let counters = zeroSnapshot();
  return {
    add: (counter, amount) => {
      counters[counter] += amount;
    },
    enabled: true,
    increment: (counter) => {
      counters[counter] += 1;
    },
    reset: () => {
      counters = zeroSnapshot();
    },
    snapshot: () => Object.freeze({ ...counters }),
  };
};
