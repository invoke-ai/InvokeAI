export interface CanvasEditIdentity {
  readonly kind: string;
  readonly layerId?: string;
}

export interface CanvasEditLease {
  readonly signal: AbortSignal;
  isCurrent(): boolean;
  release(): void;
}

export interface CanvasEditGate {
  tryAcquire(identity: CanvasEditIdentity): CanvasEditLease | null;
}

export interface CanvasEditGateController extends CanvasEditGate {
  activate(): void;
  cooldown(): void;
  invalidateDocument(): void;
  invalidateProject(): void;
  invalidateLayer(layerId: string): void;
  dispose(): void;
}

interface ActiveLease {
  readonly controller: AbortController;
  readonly generation: number;
  readonly identity: CanvasEditIdentity;
}

export const createCanvasEditGate = (): CanvasEditGateController => {
  let active: ActiveLease | null = null;
  let generation = 0;
  let lifecycle: 'active' | 'cooling' | 'disposed' = 'active';

  const invalidate = (): void => {
    generation += 1;
    const lease = active;
    active = null;
    lease?.controller.abort();
  };

  const tryAcquire = (_identity: CanvasEditIdentity): CanvasEditLease | null => {
    if (lifecycle !== 'active' || active) {
      return null;
    }
    const lease: ActiveLease = { controller: new AbortController(), generation, identity: _identity };
    active = lease;
    const isCurrent = (): boolean =>
      lifecycle === 'active' && active === lease && generation === lease.generation && !lease.controller.signal.aborted;
    return {
      isCurrent,
      release: () => {
        if (active === lease) {
          invalidate();
        }
      },
      signal: lease.controller.signal,
    };
  };

  return {
    activate: () => {
      if (lifecycle !== 'disposed') {
        lifecycle = 'active';
      }
    },
    cooldown: () => {
      if (lifecycle === 'disposed') {
        return;
      }
      lifecycle = 'cooling';
      invalidate();
    },
    dispose: () => {
      if (lifecycle === 'disposed') {
        return;
      }
      lifecycle = 'disposed';
      invalidate();
    },
    invalidateDocument: invalidate,
    invalidateLayer: (layerId) => {
      if (active?.identity.layerId === layerId) {
        invalidate();
      }
    },
    invalidateProject: invalidate,
    tryAcquire,
  };
};
