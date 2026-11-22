import type * as LayoutPresetActivationModule from '@workbench/layoutPresetActivation';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { createWorkbenchStore, type WorkbenchInternalStore, type WorkbenchSnapshot } from '@workbench/workbenchStore';
import { useCustomizeSession } from '@workbench/workspace/useStructureLock';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

let store: WorkbenchInternalStore;

vi.mock('@workbench/WorkbenchContext', () => {
  type Equality<Selected> = (left: Selected, right: Selected) => boolean;
  const useWorkbenchSelector = <Selected,>(
    selector: (snapshot: WorkbenchSnapshot) => Selected,
    isEqual: Equality<Selected> = shallowEqual
  ) => useExternalStoreSelector(store.subscribe, store.getSnapshot, selector, isEqual);

  return {
    useActiveWorkspaceId: () => useWorkbenchSelector((snapshot) => snapshot.activeProject.workspaceId),
    useActiveLayoutSelector: <Selected,>(
      selector: (layout: WorkbenchSnapshot['activeLayout']) => Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector((snapshot) => selector(snapshot.activeLayout), isEqual),
    useActiveProjectSelector: <Selected,>(
      selector: (project: WorkbenchSnapshot['activeProject']) => Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector((snapshot) => selector(snapshot.activeProject), isEqual),
    useDebouncedWorkbenchSelector: <Selected,>(selector: (snapshot: WorkbenchSnapshot) => Selected) =>
      useWorkbenchSelector(selector),
    useOptionalWorkbenchSelector: <Selected,>(
      selector: (snapshot: WorkbenchSnapshot) => Selected,
      _fallback: Selected,
      isEqual?: Equality<Selected>
    ) => useWorkbenchSelector(selector, isEqual),
    useWorkbenchCommands: () => store.commands,
    useWorkbenchQueries: () => store.queries,
    useWorkbenchSelector,
  };
});

vi.mock('@workbench/layoutPresetActivation', async (importOriginal) => ({
  ...(await importOriginal<typeof LayoutPresetActivationModule>()),
  loadLayoutWidgets: () => Promise.resolve(),
  preloadLayoutWidgets: () => undefined,
}));

vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({
  useShortcutBinding: () => null,
  useShortcutLabel: () => null,
}));

const { CustomizeBar } = await import('./CustomizeBar');
const { LayoutPresetStrip } = await import('./LayoutPresetStrip');

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The top bar's swap: the bar in the tabs' place while the active project customizes. */
const WorkspaceStrip = () => {
  const session = useCustomizeSession();

  return session ? <CustomizeBar session={session} /> : <LayoutPresetStrip />;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const bar = () => document.querySelector<HTMLElement>('[data-customize-bar]');
const composeTab = () => document.querySelector<HTMLElement>('[role="tab"][data-layout-preset-id="compose"]');
const snapshot = () => store.getSnapshot();
const rightRail = () => snapshot().activeLayout.widgetRegions.right.instanceIds;
const removeQueue = () => act(() => store.commands.widgets.toggle({ region: 'right', widgetId: 'queue' }));
const rename = async (name: string) => {
  const input = page.getByRole('textbox', { name: 'Workspace name' });

  await act(() => input.clear());
  await act(() => input.fill(name));
};

beforeEach(async () => {
  accountLifecycle.activate('customize-bar-test');
  store = createWorkbenchStore();
  store.commands.layout.applyPreset('compose');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <WorkspaceStrip />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
});

describe('Customize bar', () => {
  it('takes focus from the tab menu, and Done saves the arrangement and name, then hands focus back', async () => {
    await act(() => userEvent.click(composeTab()!, { button: 'right' }));
    await act(() => page.getByRole('menuitem', { name: 'Customize' }).click());

    await expect.poll(() => document.activeElement).toBe(bar());
    await removeQueue();
    await rename('Writing room');
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(bar()).toBeNull();
    expect(snapshot().account.layoutPresetOverrides?.compose?.widgetRegions.right.instanceIds).not.toContain('queue');
    expect(snapshot().account.layoutPresetMetadataOverrides?.compose?.label).toBe('Writing room');
    await expect.poll(() => document.activeElement).toBe(composeTab());
  });

  it('Cancel lays out what was on screen at entry and keeps the saved name', async () => {
    await act(() => store.commands.layout.beginCustomize());
    await removeQueue();
    await rename('Scratch');
    await act(() => page.getByRole('button', { name: 'Cancel' }).click());

    expect(bar()).toBeNull();
    expect(rightRail()).toContain('queue');
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(composeTab()).toHaveAccessibleName('Compose');
  });

  it('keeps an untouched built-in following its shipped default on Done, though its view had changed', async () => {
    await act(() => store.commands.layout.setRegionSize('right', 401));
    await act(() => store.commands.layout.beginCustomize());
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(snapshot().account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(snapshot().account.layoutPresetMetadataOverrides?.compose).toBeUndefined();
    expect(snapshot().account.layoutPresetRouteOverrides?.compose).toBeUndefined();
  });

  it('holds Reset to default until Done; Cancel keeps the customized built-in', async () => {
    const customizeCompose = async () => {
      await act(() => store.commands.layout.beginCustomize());
      await removeQueue();
      await rename('Writing');
      await act(() => page.getByRole('button', { name: 'Done' }).click());
      await act(() => store.commands.layout.beginCustomize());
      await act(() => page.getByRole('button', { name: 'Reset to default' }).click());
    };

    await customizeCompose();
    await expect.element(page.getByRole('textbox', { name: 'Workspace name' })).toHaveValue('Compose');
    expect(rightRail()).toContain('queue');
    await act(() => page.getByRole('button', { name: 'Cancel' }).click());

    expect(rightRail()).not.toContain('queue');
    expect(snapshot().account.layoutPresetMetadataOverrides?.compose?.label).toBe('Writing');
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeDefined();

    await act(() => store.commands.layout.beginCustomize());
    await act(() => page.getByRole('button', { name: 'Reset to default' }).click());
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(snapshot().account.layoutPresetMetadataOverrides?.compose).toBeUndefined();
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeUndefined();
    expect(composeTab()).toHaveAccessibleName('Compose');
  });

  it('keeps a reset pending only while its step is undoable: Undo withdraws it, Redo brings it back', async () => {
    await act(() => store.commands.layout.beginCustomize());
    await removeQueue();
    await rename('Writing');
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    const resetThenUndo = async () => {
      await act(() => store.commands.layout.beginCustomize());
      await act(() => page.getByRole('button', { name: 'Reset to default' }).click());
      await act(() => page.getByRole('button', { name: 'Undo' }).click());
    };

    await resetThenUndo();
    await expect.element(page.getByRole('textbox', { name: 'Workspace name' })).toHaveValue('Writing');
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(snapshot().account.layoutPresetMetadataOverrides?.compose?.label).toBe('Writing');
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeDefined();

    await resetThenUndo();
    await act(() => page.getByRole('button', { name: 'Redo' }).click());
    await expect.element(page.getByRole('textbox', { name: 'Workspace name' })).toHaveValue('Compose');
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(snapshot().account.layoutPresetMetadataOverrides?.compose).toBeUndefined();
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeUndefined();
  });

  it('returns focus to the active tab when the workspace being customized is deleted', async () => {
    await act(() => store.commands.layout.beginCustomize({ defaultRoute: null, id: 'mine', label: 'Mine' }));
    await act(() => store.commands.layout.finishCustomize());
    await act(() =>
      userEvent.click(document.querySelector<HTMLElement>('[role="tab"][data-layout-preset-id="mine"]')!, {
        button: 'right',
      })
    );
    await act(() => page.getByRole('menuitem', { name: 'Customize' }).click());
    await expect.poll(() => document.activeElement).toBe(bar());

    await act(() => store.commands.layout.deletePreset('mine'));

    expect(bar()).toBeNull();
    await expect.poll(() => document.activeElement).toBe(composeTab());
  });

  it('drafts New workspace with its name focused: Cancel adds no tab and Done adds one, the project on it', async () => {
    const tabCount = () => document.querySelectorAll('[role="tab"][data-layout-preset-id]').length;
    const name = page.getByRole('textbox', { name: 'Workspace name' });
    const account = snapshot().account;

    await act(() => page.getByRole('button', { name: 'New workspace' }).click());
    await expect.poll(() => document.activeElement).toBe(name.element());
    await expect.element(name).toHaveValue('Compose copy');
    await act(() => page.getByRole('button', { name: 'Cancel' }).click());

    expect(snapshot().account).toBe(account);
    expect(tabCount()).toBe(4);

    await act(() => page.getByRole('button', { name: 'New workspace' }).click());
    await rename('Review');
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    expect(tabCount()).toBe(5);
    await expect.element(page.getByRole('tab', { name: 'Review' })).toHaveAttribute('aria-selected', 'true');
    expect(snapshot().account.layoutPresetOverrides?.compose).toBeUndefined();
  });

  it('keeps Cancel and Done in view in a narrow window, with a cue for the controls scrolled away', async () => {
    host!.style.width = '420px';
    await act(() => store.commands.layout.beginCustomize());
    const done = page.getByRole('button', { name: 'Done' }).element();

    expect(done.getBoundingClientRect().right).toBeLessThanOrEqual(bar()!.getBoundingClientRect().right);
    await expect.element(page.getByRole('button', { name: 'Scroll to end' })).toBeInTheDocument();
  });
});
