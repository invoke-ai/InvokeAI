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

// The store is read for real but never written: `activatePreset` is a stub the
// test can freeze outright, which is the point of the suite. If the tab only
// acknowledges the press once the store has caught up, nothing here passes.
let store: WorkbenchInternalStore;
let activatePreset: (presetId: LayoutPresetId) => Promise<void> = () => Promise.resolve();

vi.mock('@workbench/WorkbenchContext', () => {
  const useSnapshot = () => useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  const noop = () => undefined;

  return {
    shallowEqual: Object.is,
    useActiveProjectSelector: <Selected,>(
      selector: (project: ReturnType<typeof useSnapshot>['activeProject']) => Selected
    ) => selector(useSnapshot().activeProject),
    useDebouncedWorkbenchSelector: <Selected,>(selector: (snapshot: ReturnType<typeof useSnapshot>) => Selected) =>
      selector(useSnapshot()),
    useWorkbenchCommands: () => ({
      layout: {
        activatePreset: (presetId: LayoutPresetId) => activatePreset(presetId),
        createPreset: noop,
        reorderPresets: noop,
        reset: noop,
        savePreset: noop,
      },
    }),
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

const render = async (options: {
  activatePreset: (presetId: LayoutPresetId) => Promise<void>;
  activePresetId: LayoutPresetId;
}) => {
  activatePreset = options.activatePreset;
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

const press = (element: Element): Promise<void> =>
  act(async () => {
    element.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));
    await Promise.resolve();
  });

const nextFrame = (): Promise<void> =>
  act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => resolve(null));
    });
  });

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  activatePreset = () => Promise.resolve();
});

describe('LayoutPresetStrip acknowledgment', () => {
  it('marks the pressed tab active on pointerdown, before any store update', async () => {
    // activatePreset never resolves: the store is deliberately frozen.
    const activate = vi.fn(() => new Promise<void>(() => {}));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    const editTab = presetTab('edit');

    expect(editTab.hasAttribute('data-preset-active')).toBe(false);

    await press(editTab);

    expect(presetTab('edit').hasAttribute('data-preset-active')).toBe(true);
    expect(presetTab('compose').hasAttribute('data-preset-active')).toBe(false);
    expect(store.getSnapshot().activeProject.layout.presetId).toBe('compose');
  });

  // The performance gate watches `aria-selected`, and assistive technology reads
  // it: the acknowledgment is only real if the tabs machine moves with it.
  it('moves the tabs machine selection in the same commit as the acknowledgment', async () => {
    const activate = vi.fn(() => new Promise<void>(() => {}));

    await render({ activatePreset: activate, activePresetId: 'compose' });

    await press(presetTab('edit'));

    expect(presetTab('edit').getAttribute('aria-selected')).toBe('true');
    expect(presetTab('compose').getAttribute('aria-selected')).toBe('false');
    expect(presetTab('compose').hasAttribute('data-selected')).toBe(false);
  });

  it('activates on the frame after the acknowledgment, not inside the handler', async () => {
    const activate = vi.fn(() => Promise.resolve());

    await render({ activatePreset: activate, activePresetId: 'compose' });

    const editTab = presetTab('edit');

    await act(async () => {
      editTab.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));

      // Checked before yielding: the handler itself must not have dispatched.
      expect(activate).not.toHaveBeenCalled();
      await Promise.resolve();
    });

    await nextFrame();

    expect(activate).toHaveBeenCalledWith('edit');
  });

  it('does not re-activate the preset that is already active', async () => {
    const activate = vi.fn(() => Promise.resolve());

    await render({ activatePreset: activate, activePresetId: 'compose' });

    await press(presetTab('compose'));
    await nextFrame();

    expect(activate).not.toHaveBeenCalled();
  });
});
