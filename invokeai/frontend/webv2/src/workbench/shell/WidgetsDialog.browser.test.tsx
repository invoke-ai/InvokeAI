import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

const workbench = vi.hoisted(() => ({ store: null as ReturnType<typeof createWorkbenchStore> | null }));

// A real store without the provider's persistence: the list reads placement from it and acts through its commands.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');

  return {
    ...(await importOriginal<typeof WorkbenchContextModule>()),
    useActiveProjectSelector: <Selected,>(selector: (project: unknown) => Selected) =>
      selector(useSyncExternalStore(workbench.store!.subscribe, () => workbench.store!.getSnapshot().activeProject)),
    useWorkbenchCommands: () => workbench.store!.commands,
    useWorkbenchQueries: () => workbench.store!.queries,
  };
});

const { WidgetsDialog } = await import('./WidgetsDialog');
const { closeWidgetsDialog, openWidgetsDialog, openWidgetsDialogFrom } = await import('./widgetsDialogStore');

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const store = () => workbench.store!;
const activeProject = () => store().getSnapshot().activeProject;
const row = (key: string) => page.getByRole('dialog').element().querySelector(`[data-widgets-row="${key}"]`);
const sectionItems = (name: string) => page.getByRole('list', { name }).getByRole('listitem').elements();

beforeEach(async () => {
  accountLifecycle.activate('widgets-dialog-test');
  workbench.store = createWorkbenchStore();
  store().commands.layout.applyPreset('compose');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
            <WidgetsDialog />
          </WorkbenchWidgetRegistryProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  await act(() => openWidgetsDialog());
  await expect.element(page.getByRole('dialog', { name: 'Widgets' })).toBeVisible();
});

afterEach(async () => {
  await act(() => closeWidgetsDialog());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  workbench.store = null;
  accountLifecycle.invalidate();
});

describe('WidgetsDialog', () => {
  it('lists placed widgets with where they are, and brings a removed one back', async () => {
    // Compose docks the Queue on the right and leaves Layers out.
    expect(sectionItems('In this layout')).toContain(row('queue'));
    expect(row('queue')?.textContent).toContain('Right panel');
    expect(sectionItems('Not in this layout')).toContain(row('type:layers'));

    await act(() => page.getByRole('button', { name: 'Remove Queue (Right panel)' }).click());

    expect(activeProject().widgetRegions.right.instanceIds).not.toContain('queue');
    expect(sectionItems('Not in this layout')).toContain(row('type:queue'));
    // The row's buttons changed under focus; focus lands on what the list offers for that widget now.
    await expect.element(page.getByRole('button', { name: 'Add Queue' })).toHaveFocus();

    await act(() => page.getByRole('button', { name: 'Add Queue' }).click());
    expect(row('type:queue')).toBeNull();
  });

  // Status chips render no panel, so a placement open refuses them; the list puts them back in the status bar.
  it('adds a removed status-bar chip back to the status bar', async () => {
    const autosave = activeProject().widgetRegions.bottom.instanceIds.find(
      (instanceId) => activeProject().widgetInstances[instanceId]?.typeId === 'autosave-status'
    )!;

    await act(() => page.getByRole('button', { name: 'Remove Autosave (Status bar)' }).click());
    expect(activeProject().widgetRegions.bottom.instanceIds).not.toContain(autosave);
    await expect.element(page.getByRole('button', { name: 'Add Autosave' })).toHaveFocus();

    await act(() => page.getByRole('button', { name: 'Add Autosave' }).click());

    expect(
      activeProject().widgetRegions.bottom.instanceIds.some(
        (instanceId) => activeProject().widgetInstances[instanceId]?.typeId === 'autosave-status'
      )
    ).toBe(true);
  });

  it('keeps the last center view, saying why its Remove is unavailable', async () => {
    await act(() => page.getByRole('button', { name: 'Remove Gallery (Center)' }).click());
    const center = [...activeProject().widgetRegions.center.instanceIds];
    const remove = page.getByRole('button', { name: 'Remove Preview (Center)' });

    await expect.element(remove).toHaveAttribute('aria-disabled', 'true');
    await expect.element(remove).toHaveAccessibleDescription('The center always shows one view.');

    // A pointer click would be refused as disabled; dispatch one to prove the action itself is blocked.
    await act(() => (remove.element() as HTMLElement).click());

    expect(activeProject().widgetRegions.center.instanceIds).toEqual(center);
    await expect.element(page.getByRole('dialog', { name: 'Widgets' })).toBeVisible();
  });

  it('lists rows that read the same in one order: by where they are', () => {
    const order = ['Center', 'Left panel', 'Right panel', 'Status bar'];
    const galleryPlaces = [...page.getByRole('dialog').element().querySelectorAll('[data-widgets-type="gallery"]')].map(
      (row) => order.find((place) => row.textContent?.includes(place)) ?? ''
    );

    expect(galleryPlaces.length).toBeGreaterThan(1);
    expect(galleryPlaces).toEqual([...galleryPlaces].sort((left, right) => order.indexOf(left) - order.indexOf(right)));
  });

  it('returns focus on close to the control its opener named, though the menu item that opened it is gone', async () => {
    const trigger = document.createElement('button');
    trigger.textContent = 'Menu trigger';
    host!.append(trigger);
    await act(() => closeWidgetsDialog());
    await expect.element(page.getByRole('dialog', { name: 'Widgets' })).not.toBeInTheDocument();

    await act(() => openWidgetsDialogFrom(() => trigger));
    await expect.element(page.getByRole('dialog', { name: 'Widgets' })).toBeVisible();
    await act(() => page.getByRole('button', { name: 'Done' }).click());

    await expect.element(trigger).toHaveFocus();
  });

  it('docks a floating window from the list', async () => {
    await act(() => store().commands.widgets.float('queue'));
    expect(row('queue')?.textContent).toContain('Floating window');

    await act(() => page.getByRole('button', { name: 'Dock Queue (Floating window)' }).click());

    expect(activeProject().floatingWidgets?.queue).toBeUndefined();
    expect(row('queue')?.textContent).toContain('Right panel');
  });

  it('closes when a widget is shown, so focus can go to it', async () => {
    await act(() => page.getByRole('button', { name: 'Show Queue (Right panel)' }).click());

    await expect.element(page.getByRole('dialog', { name: 'Widgets' })).not.toBeInTheDocument();
  });
});
