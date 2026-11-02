import type { Project } from '@workbench/projectContracts';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import { subscribeTrackpadDetection } from './CanvasTrackpadDetection';
import { recordTrackpadNavigationChoice } from './trackpadNavigation';

type Preferences = Pick<WorkbenchPreferences, 'canvasTrackpadNavigationDecided' | 'scopedDefaults'>;

/**
 * The real store persists to local storage and the backend (its saved-state read is covered in `store.test.ts`). This
 * account keeps the two copies that matter here apart: this window's snapshot and the saved preferences, which another
 * window may have changed.
 */
const account = vi.hoisted(() => ({
  /** Holds the saved-state write until released, standing in for the backend round trip. */
  held: null as Promise<void> | null,
  hydrated: true,
  saved: {} as Preferences,
  snapshot: {} as Preferences,
}));
vi.mock('@workbench/settings/store', () => ({
  areWorkbenchPreferencesHydrated: () => account.hydrated,
  getWorkbenchPreferences: () => account.snapshot,
  patchWorkbenchPreferences: (patch: Partial<Preferences>) => {
    account.snapshot = { ...account.snapshot, ...patch };
    account.saved = { ...account.saved, ...patch };
    return Promise.resolve();
  },
  updateSavedWorkbenchPreferences: async (update: (saved: Preferences) => Partial<Preferences> | null) => {
    await account.held;
    const patch = update(account.saved);
    account.saved = { ...account.saved, ...patch };
    account.snapshot = account.saved;
    return patch !== null;
  },
}));

const undecided = (): Preferences => ({ canvasTrackpadNavigationDecided: false, scopedDefaults: { canvas: {} } });

const createHarness = (canvasPins: Record<string, unknown> = {}) => {
  const project = {
    widgetInstances: {
      canvas: {
        id: 'canvas',
        state: { id: 'canvas', label: 'Canvas', values: canvasPins, version: 1 },
        typeId: 'canvas',
      },
    },
  } as unknown as Project;
  const engineState = { trackpadNavigation: false };
  let pinchListener: (() => void) | null = null;
  const engine = {
    interaction: {
      set: (key: string, value: boolean) => {
        if (key === 'trackpadNavigation') {
          engineState.trackpadNavigation = value;
        }
      },
    },
    tools: {
      onTrackpadPinch: (listener: () => void) => {
        pinchListener = listener;
        return () => {
          pinchListener = null;
        };
      },
    },
  };
  const queries = {
    getProject: (id: string) => (id === 'p1' ? project : null),
    getSnapshot: () => ({ scopedDefaults: account.snapshot.scopedDefaults }) as WorkbenchSnapshot,
  };
  const unsubscribe = subscribeTrackpadDetection(engine as never, queries, 'p1');
  return { engineState, pinch: () => pinchListener?.(), unsubscribe };
};

beforeEach(() => {
  account.held = null;
  account.hydrated = true;
  account.saved = undecided();
  account.snapshot = undecided();
});

describe('trackpad navigation auto-enable', () => {
  it('turns the engine setting on before the pinch is routed and saves it as the user default', async () => {
    account.saved = { ...account.saved, scopedDefaults: { canvas: { showGrid: true } } };
    const h = createHarness();
    h.pinch();
    // Synchronously, so the very pinch that was detected zooms.
    expect(h.engineState.trackpadNavigation).toBe(true);
    await vi.waitFor(() => expect(account.saved.canvasTrackpadNavigationDecided).toBe(true));
    // Only the flag and the one key, over the saved defaults.
    expect(account.saved.scopedDefaults).toEqual({ canvas: { showGrid: true, trackpadNavigation: true } });
  });

  it('never overrides an explicit choice', () => {
    recordTrackpadNavigationChoice();
    const h = createHarness();
    h.pinch();
    expect(h.engineState.trackpadNavigation).toBe(false);
    expect(account.saved.scopedDefaults).toEqual({ canvas: {} });
  });

  it('defers to a choice saved in another window, though this window has not seen it', async () => {
    account.saved = { canvasTrackpadNavigationDecided: true, scopedDefaults: { canvas: {} } };
    const h = createHarness();
    h.pinch();
    expect(h.engineState.trackpadNavigation).toBe(true);
    // Back to the project's value once the saved state says it was decided.
    await vi.waitFor(() => expect(h.engineState.trackpadNavigation).toBe(false));
    expect(account.snapshot.canvasTrackpadNavigationDecided).toBe(true);
    expect(account.saved.scopedDefaults).toEqual({ canvas: {} });
  });

  it('reverts the engine even when the subscriber is gone before the saved state answers', async () => {
    let release = () => {};
    account.held = new Promise<void>((resolve) => {
      release = resolve;
    });
    account.saved = { canvasTrackpadNavigationDecided: true, scopedDefaults: { canvas: {} } };
    const h = createHarness();
    h.pinch();
    expect(h.engineState.trackpadNavigation).toBe(true);
    // Adopting the decision disarms detection, which unmounts the subscriber before the write resolves.
    h.unsubscribe();
    release();
    await vi.waitFor(() => expect(h.engineState.trackpadNavigation).toBe(false));
  });

  it('leaves a project that pins the setting alone, without recording a decision', () => {
    const h = createHarness({ trackpadNavigation: false });
    h.pinch();
    expect(h.engineState.trackpadNavigation).toBe(false);
    expect(account.saved.canvasTrackpadNavigationDecided).toBe(false);
  });

  it('does nothing before the account preferences have loaded', () => {
    account.hydrated = false;
    const h = createHarness();
    h.pinch();
    recordTrackpadNavigationChoice();
    expect(h.engineState.trackpadNavigation).toBe(false);
    expect(account.saved).toEqual(undecided());
  });
});
