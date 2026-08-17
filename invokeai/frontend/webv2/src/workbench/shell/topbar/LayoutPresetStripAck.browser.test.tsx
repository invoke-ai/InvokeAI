import type { LayoutPresetId } from '@workbench/layoutContracts';
import type * as LayoutPresetActivationModule from '@workbench/layoutPresetActivation';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createWorkbenchStore, type WorkbenchInternalStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

type ActivatePreset = (presetId: LayoutPresetId) => Promise<LayoutPresetId | null>;

// The store is read for real but never written: `activatePreset` is a stub the
// test can freeze outright, which is the point of the suite. If the tab only
// acknowledges the press once the store has caught up, nothing here passes.
let store: WorkbenchInternalStore;

// Rebuilt per render rather than read from a module slot at call time. The
// strip dispatches activation from a `requestAnimationFrame`, so a frame that
// outlives its test would otherwise reach into the *next* test's spy and
// satisfy its assertions with a stale call.
const noop = () => undefined;
let commands = { layout: { activatePreset: (() => Promise.resolve(null)) as ActivatePreset } };

const createCommands = (activatePreset: ActivatePreset) => ({
  layout: { activatePreset, createPreset: noop, reorderPresets: noop, reset: noop, savePreset: noop },
});

vi.mock('@workbench/WorkbenchContext', () => {
  const useSnapshot = () => useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  return {
    shallowEqual: Object.is,
    useActiveProjectSelector: <Selected,>(
      selector: (project: ReturnType<typeof useSnapshot>['activeProject']) => Selected
    ) => selector(useSnapshot().activeProject),
    useDebouncedWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<typeof useSnapshot>) => Selected) =>
      selector(useSnapshot()),
    useWorkbenchCommands: () => commands,
    useWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<typeof useSnapshot>) => Selected) =>
      selector(useSnapshot()),
  };
});

vi.mock('@workbench/layoutPresetActivation', async (importOriginal) => ({
  ...(await importOriginal<typeof LayoutPresetActivationModule>()),
  preloadLayoutPresetWidgets: () => undefined,
}));
vi.mock('./useTopbarShortcut', () => ({ useTopbarShortcut: () => null }));

import { LayoutPresetStrip } from './LayoutPresetStrip';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const nextFrame = (): Promise<void> =>
  act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => resolve(null));
    });
  });

const render = async (options: { activatePreset: ActivatePreset; activePresetId: LayoutPresetId }) => {
  commands = createCommands(options.activatePreset);
  store = createWorkbenchStore();
  store.commands.layout.applyPreset(options.activePresetId);
  host = document.createElement('div');
  host.style.width = '900px';
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <LayoutPresetStrip />
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const presetTab = (id: string): HTMLElement => {
  const tab = document.querySelector<HTMLElement>(`[role="tab"][data-layout-preset-id="${id}"]`);

  if (!tab) {
    throw new Error(`No preset tab rendered for ${id}.`);
  }

  return tab;
};

const press = (element: Element, button = 0): Promise<void> =>
  act(async () => {
    element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button, cancelable: true }));
    await Promise.resolve();
  });

afterEach(async () => {
  // Drained while the tree that queued it is still mounted, so no activation
  // frame can leak into the next test.
  await nextFrame();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  commands = createCommands(() => Promise.resolve(null));
});

describe('LayoutPresetStrip acknowledgment', () => {
  // The selected paint, `aria-selected` for assistive technology, and the mark
  // the performance gate observes are all the same attribute pair, written by
  // the tabs machine from the controlled `value`. Pinning it here means a
  // future release that defers those writes fails a test rather than silently
  // regressing the budget.
  it('selects the pressed tab on pointerdown, before any store update', async () => {
    // activatePreset never resolves: the store is deliberately frozen.
    const activate = vi.fn<ActivatePreset>(() => new Promise<LayoutPresetId | null>(() => {}));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    expect(presetTab('edit').getAttribute('aria-selected')).toBe('false');

    await press(presetTab('edit'));

    expect(presetTab('edit').getAttribute('aria-selected')).toBe('true');
    expect(presetTab('edit').hasAttribute('data-selected')).toBe(true);
    expect(presetTab('compose').getAttribute('aria-selected')).toBe('false');
    expect(presetTab('compose').hasAttribute('data-selected')).toBe(false);
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
  });

  it('activates on the frame after the acknowledgment, not inside the handler', async () => {
    const activate = vi.fn<ActivatePreset>(() => Promise.resolve('edit'));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    const editTab = presetTab('edit');

    await act(async () => {
      editTab.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, cancelable: true }));

      // Checked before yielding: the handler itself must not have dispatched.
      expect(activate).not.toHaveBeenCalled();
      await Promise.resolve();
    });

    await nextFrame();

    expect(activate).toHaveBeenCalledWith('edit');
  });

  it('does not re-activate the preset that is already active', async () => {
    const activate = vi.fn<ActivatePreset>(() => Promise.resolve('compose'));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    await press(presetTab('compose'));
    await nextFrame();

    expect(activate).not.toHaveBeenCalled();
  });

  // `pointerdown` fires for the secondary button too, ahead of `contextmenu`.
  // Switching on right-click would strip the menu of the "switch to this
  // preset" item that is the whole reason to open it on an inactive tab.
  it('ignores a secondary-button press', async () => {
    const activate = vi.fn<ActivatePreset>(() => Promise.resolve('edit'));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    await press(presetTab('edit'), 2);
    await nextFrame();

    expect(activate).not.toHaveBeenCalled();
    expect(presetTab('edit').getAttribute('aria-selected')).toBe('false');
    expect(presetTab('compose').getAttribute('aria-selected')).toBe('true');
  });

  // Three activations are silently dropped by the store — superseded, overtaken
  // by a project switch, or aimed at a replaced preset definition. Painting the
  // selection ahead of the store means the tab has to be handed back when that
  // happens, or it shows a preset nothing ever applied, permanently.
  it('gives the selection back to the store when the activation is dropped', async () => {
    // Settled by hand inside `act` rather than pre-resolved: it pins the
    // handoff to the moment the outcome is known, instead of to whichever
    // microtask turn happens to win the race.
    let reportOutcome!: (appliedPresetId: LayoutPresetId | null) => void;
    const outcome = new Promise<LayoutPresetId | null>((resolve) => {
      reportOutcome = resolve;
    });
    const activate = vi.fn<ActivatePreset>(() => outcome);

    await render({ activatePreset: activate, activePresetId: 'compose' });

    await press(presetTab('edit'));

    expect(presetTab('edit').getAttribute('aria-selected')).toBe('true');

    await nextFrame();

    expect(activate).toHaveBeenCalledWith('edit');
    expect(presetTab('edit').getAttribute('aria-selected')).toBe('true');

    await act(async () => {
      reportOutcome(null);
      await outcome;
    });

    expect(presetTab('edit').getAttribute('aria-selected')).toBe('false');
    expect(presetTab('compose').getAttribute('aria-selected')).toBe('true');
  });
});
