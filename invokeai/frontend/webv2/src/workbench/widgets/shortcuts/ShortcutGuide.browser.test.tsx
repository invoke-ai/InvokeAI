import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Project } from '@workbench/projectContracts';
import type * as SettingsStoreModule from '@workbench/settings/store';
import type * as WidgetFrameModule from '@workbench/widget-frame';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, useSensor, useSensors } from '@dnd-kit/core';
import { createExternalStore } from '@platform/state/externalStore';
import { registerModalPresence } from '@platform/ui/modalPresence';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { useRegisterShortcutHintSource, type ShortcutHintSnapshot } from '@workbench/hotkeys/hintSources';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, useMemo, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const runtime = vi.hoisted(() => ({
  store: null as unknown as WorkbenchInternalStore,
  extensions: null as unknown as ExtensionRegistry,
  customHotkeys: {} as Record<string, string[]>,
  registrations: 0,
  releases: 0,
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) => {
    const state = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
    return selector(state.activeProject);
  },
  useOptionalWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchQueries: () => runtime.store.queries,
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchSubscription: () => runtime.store.subscribe,
}));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof SettingsStoreModule>()),
  useWorkbenchPreferenceSelector: <Selected,>(
    select: (preferences: { customHotkeys: Record<string, string[]>; showFocusRegionHighlight: boolean }) => Selected
  ) =>
    select({
      customHotkeys: runtime.customHotkeys,
      showFocusRegionHighlight: true,
    }),
}));
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

import { PrimaryMouseSensor } from '@workbench/shell/holdToDragSensor';
import { StatusBar } from '@workbench/shell/StatusBar';

import { ShortcutGuide } from './ShortcutGuide';

const widgetWidths = createExternalStore<Record<string, number>>({});

vi.mock('@workbench/widget-frame', async (importOriginal) => ({
  ...(await importOriginal<typeof WidgetFrameModule>()),
  WidgetRendererById: ({ instanceId, presentation }: { instanceId: string; presentation: 'compact' | 'expanded' }) => {
    const widths = useSyncExternalStore(widgetWidths.subscribe, widgetWidths.getSnapshot);
    return instanceId === 'shortcuts' ? (
      <ShortcutGuide presentation={presentation} />
    ) : (
      <div style={{ width: widths[instanceId] ?? 0 }} />
    );
  },
}));

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
let releaseModal: (() => void) | null = null;
let executed: string[] = [];
const initialHints: ShortcutHintSnapshot = {
  titleKey: 'widgets.canvas.tools.brush',
  hints: [
    { labelKey: 'workbench.shortcuts.actions.apply', parts: ['enter'] },
    { labelKey: 'workbench.shortcuts.actions.cancel', parts: ['esc'] },
    { labelKey: 'workbench.shortcuts.actions.pan', parts: ['space'], pointerKey: 'workbench.shortcuts.pointer.drag' },
    { labelKey: 'workbench.shortcuts.actions.sample', parts: ['alt'], pointerKey: 'workbench.shortcuts.pointer.click' },
  ],
};
const hints = createExternalStore(initialHints);

const Harness = ({ width = 900, withCanvasSource = true }: { width?: number; withCanvasSource?: boolean }) => {
  const sensors = useSensors(useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }));
  const projectId = runtime.store.getSnapshot().activeProject.id;
  const source = useMemo(
    () => ({
      projectId,
      instanceId: 'canvas',
      getSnapshot: hints.getSnapshot,
      subscribe: (listener: () => void) => {
        runtime.registrations++;
        const release = hints.subscribe(listener);
        return () => {
          runtime.releases++;
          release();
        };
      },
    }),
    [projectId]
  );
  useRegisterShortcutHintSource(withCanvasSource ? source : null);
  return (
    <>
      <WorkbenchHotkeyRuntime />
      <div {...useFocusRegionProps('center')}>
        <div
          data-hotkey-widget-instance-id="canvas"
          data-hotkey-widget-type-id="canvas"
          data-hotkey-widget-region="center"
          tabIndex={-1}
          data-testid="canvas"
        >
          <input aria-label="Canvas field" />
        </div>
      </div>
      <div {...useFocusRegionProps('right')}>
        <div
          data-hotkey-widget-instance-id="layers"
          data-hotkey-widget-type-id="layers"
          data-hotkey-widget-region="right"
        >
          <input aria-label="Property field" />
        </div>
      </div>
      <div {...useFocusRegionProps('bottom')} style={{ display: 'flex', height: 24, width }}>
        <DndContext sensors={sensors}>
          <StatusBar dropState={{ helperText: '', isActive: false, isAllowed: true }} />
        </DndContext>
      </div>
    </>
  );
};

const render = async (width = 900, withCanvasSource = true) => {
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <WorkbenchFocusProvider>
            <Harness width={width} withCanvasSource={withCanvasSource} />
          </WorkbenchFocusProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});
beforeEach(async () => {
  hints.setSnapshot(initialHints);
  runtime.store = createWorkbenchStore();
  runtime.extensions = createExtensionRegistry();
  runtime.registrations = 0;
  runtime.releases = 0;
  runtime.customHotkeys = {
    'app.openCommandPalette': ['mod+j'],
    'app.invoke': ['mod+shift+j'],
    'app.cancelQueueItem': ['mod+shift+x'],
  };
  widgetWidths.setSnapshot({});
  executed = [];
  for (const id of ['app.invoke', 'app.cancelQueueItem', 'app.openCommandPalette']) {
    runtime.extensions.commands.register({
      handler: () => {
        executed.push(id);
      },
      id,
      title: id,
    });
  }
  runtime.store.commands.layout.applyPreset('edit');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await page.viewport(1200, 800);
  await render();
});
afterEach(async () => {
  releaseModal?.();
  releaseModal = null;
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('contextual status-bar guide', () => {
  it('advertises usable remap alternatives while keeping native Enter/Space activation and Escape dismissal', async () => {
    runtime.customHotkeys = {
      'app.invoke': ['enter', 'mod+shift+j'],
      'app.cancelQueueItem': ['space'],
      'app.openCommandPalette': ['esc'],
    };
    await act(() => root!.unmount());
    root = createRoot(host!);
    await render();
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    expect(guide.element().textContent).toContain('Invoke');
    expect(guide.element().textContent).not.toContain('Cancel generation');
    expect(guide.element().textContent).not.toContain('Command palette');
    const parts = [...guide.element().querySelectorAll('[data-shortcut-guide] > :last-child kbd')].map(
      (element) => element.textContent
    );
    expect(parts).toEqual(['ctrl', 'shift', 'j']);
    await act(() => (guide.element() as HTMLElement).focus());
    await userEvent.keyboard('{Control>}{Shift>}j{/Shift}{/Control}');
    await expect.poll(() => executed).toEqual(['app.invoke']);
    await userEvent.keyboard('{Enter}');
    await expect.element(page.getByRole('dialog')).toBeVisible();
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    await act(() => (guide.element() as HTMLElement).focus());
    await userEvent.keyboard(' ');
    await expect.element(page.getByRole('dialog')).toBeVisible();
    expect(executed).toEqual(['app.invoke']);
  });

  it('keeps a meaningful settings route when all global bindings are unassigned', async () => {
    runtime.customHotkeys = { 'app.openCommandPalette': [], 'app.invoke': [], 'app.cancelQueueItem': [] };
    await act(() => root!.unmount());
    root = createRoot(host!);
    await render();
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    await guide.click();
    await expect
      .poll(() => page.getByRole('dialog').element().textContent)
      .toContain('No keyboard shortcuts are assigned for this context.');
    await expect.element(page.getByRole('button', { exact: true, name: 'Keyboard shortcut settings' })).toBeVisible();
  });

  it('registers a source once through context and hint updates and releases it on unmount', async () => {
    expect(runtime.registrations).toBe(1);
    await page.getByRole('textbox', { name: 'Property field' }).click();
    await act(() =>
      hints.setSnapshot({
        titleKey: 'widgets.canvas.tools.view',
        hints: [
          { labelKey: 'workbench.shortcuts.actions.pan', parts: [], pointerKey: 'workbench.shortcuts.pointer.drag' },
        ],
      })
    );
    await render(700);
    expect(runtime.registrations).toBe(1);
    expect(runtime.releases).toBe(0);
    await act(() => root!.unmount());
    root = null;
    expect(runtime.releases).toBe(1);
  });
  it('fits remaining footer space after other widgets and keeps Notifications visible when their labels grow', async () => {
    await act(() =>
      widgetWidths.setSnapshot({
        'server-status': 220,
        'queue-status': 100,
        'gallery:bottom': 60,
        'autosave-status': 80,
        notifications: 128,
      })
    );
    await page.viewport(700, 600);
    await render(700);
    const footer = host!.querySelector('footer')!;
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    const notification = host!.querySelector('[data-status-bar-widget="notifications"]')!;
    await expect.poll(() => footer.scrollWidth).toBeLessThanOrEqual(footer.clientWidth);
    await expect.poll(() => notification.getBoundingClientRect().right).toBeLessThanOrEqual(700);
    expect(guide.element().getBoundingClientRect().width).toBeLessThan(80);
    await act(() => widgetWidths.setSnapshot({ ...widgetWidths.getSnapshot(), 'server-status': 250 }));
    await render(700);
    await expect.poll(() => guide.element().getBoundingClientRect().width).toBeLessThanOrEqual(24);
    expect(guide.element().querySelector('[data-shortcut-guide]')?.lastElementChild?.textContent).toBe('');
    expect(notification.getBoundingClientRect().right).toBeLessThanOrEqual(700);
    await guide.click();
    await expect.element(page.getByRole('dialog')).toBeVisible();
    await expect.poll(() => page.getByRole('dialog').element().textContent).toContain('Invoke');
    await userEvent.keyboard('{Escape}');
    await act(() =>
      runtime.store.commands.widgets.setAlignment({ align: 'start', instanceId: 'shortcuts', region: 'bottom' })
    );
    await expect.poll(() => footer.scrollWidth).toBeLessThanOrEqual(footer.clientWidth);
    expect(notification.getBoundingClientRect().right).toBeLessThanOrEqual(700);
  });
  it('shrinks whole hints, preserves source in its popover, consumes Escape and restores the editing origin', async () => {
    const canvas = host!.querySelector<HTMLElement>('[data-testid="canvas"]')!;
    await act(() => canvas.focus());
    await expect.poll(() => host!.textContent).toContain('Brush');
    await page.viewport(220, 600);
    await render(220);
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    await expect
      .poll(() => host!.querySelector('[data-hint-measurements]')?.nextElementSibling?.textContent)
      .toBe('Shortcuts');
    await guide.click();
    const popup = page.getByRole('dialog');
    await expect.element(popup).toBeVisible();
    await expect.poll(() => popup.element().textContent).toContain('Brush');
    await expect.poll(() => popup.element().textContent).toContain('Apply transform');
    const popupRect = popup.element().getBoundingClientRect();
    expect(popupRect.width).toBeLessThanOrEqual(window.innerWidth - 16);
    expect(popupRect.left).toBeGreaterThanOrEqual(0);
    let reachedEngine = 0;
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        reachedEngine++;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    try {
      await userEvent.keyboard('{Escape}');
      await expect.element(popup).not.toBeInTheDocument();
      expect(reachedEngine).toBe(0);
      await expect.poll(() => document.activeElement).toBe(canvas);
    } finally {
      window.removeEventListener('keydown', onKeyDown);
    }
  });

  it('labels properties guidance, keeps it through editable focus, hides beneath modals and forgets a switched project', async () => {
    await page.getByRole('textbox', { name: 'Property field' }).click();
    await page.getByRole('button', { name: 'Shortcuts', exact: true }).click();
    await expect.poll(() => page.getByRole('dialog').element().textContent).toContain('On Canvas · Brush');
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('textbox', { name: 'Property field' })).toHaveFocus();
    await act(() => {
      releaseModal = registerModalPresence();
    });
    await expect.poll(() => host!.querySelector('[data-status-bar-widget="shortcuts"]')).toBeNull();
    await expect.element(page.getByRole('button', { name: 'Shortcuts', exact: true })).not.toBeInTheDocument();
    await act(() => {
      releaseModal!();
      releaseModal = null;
    });
    await expect.element(page.getByRole('button', { name: 'Shortcuts', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Shortcuts', exact: true }).click();
    await expect.poll(() => page.getByRole('dialog').element().textContent).toContain('On Canvas · Brush');
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('textbox', { name: 'Property field' })).toHaveFocus();
    await act(() => runtime.store.commands.projects.create());
    await expect.poll(() => host!.textContent).toContain('Invoke');
    await expect.poll(() => host!.textContent).not.toContain('On Canvas');
  });

  it('shows custom global bindings and omits background tool hints outside their owning surface', async () => {
    // The default view has no focus owner until the user enters a widget.
    await expect.poll(() => host!.textContent).toContain('Invoke');
    await expect.poll(() => host!.querySelector('kbd')?.textContent).toBe('ctrl');
    expect(host!.textContent).toContain('Cancel generation');
    expect(host!.textContent).not.toContain('Apply transform');
  });

  it('drops commands a text field blocks while it is edited and restores them once focus leaves for the page', async () => {
    const field = document.createElement('input');
    field.setAttribute('aria-label', 'Page field');
    document.body.append(field);
    try {
      await expect.poll(() => host!.textContent).toContain('Cancel generation');
      await act(() => field.focus());
      // Invoke stays usable from text fields; Cancel generation is not, so the guide must judge the focused field.
      await expect.poll(() => host!.textContent).not.toContain('Cancel generation');
      expect(host!.textContent).toContain('Invoke');
      // Leaving for the body fires focusout without a focusin.
      await act(() => field.blur());
      expect(document.activeElement).toBe(document.body);
      await expect.poll(() => host!.textContent).toContain('Cancel generation');
    } finally {
      field.remove();
    }
  });

  it('returns focus to the chip, not the page, when opened after focus left for the background', async () => {
    await page.getByRole('textbox', { name: 'Canvas field' }).click();
    await act(() => (document.activeElement as HTMLElement).blur());
    expect(document.activeElement).toBe(document.body);
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    await act(() => (guide.element() as HTMLElement).focus());
    await userEvent.keyboard('{Enter}');
    await expect.element(page.getByRole('dialog')).toBeVisible();
    await userEvent.keyboard('{Escape}');
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    await expect.poll(() => document.activeElement).toBe(guide.element());
  });

  it('keeps the global title for an edited canvas target without a mounted canvas hint source', async () => {
    await act(() => root!.unmount());
    root = createRoot(host!);
    await render(900, false);
    await page.getByRole('textbox', { name: 'Canvas field' }).click();
    await page.getByRole('button', { name: 'Shortcuts', exact: true }).click();
    const popup = page.getByRole('dialog');
    await expect.poll(() => popup.element().textContent).toContain('Workbench');
    expect(popup.element().textContent).not.toContain('On Canvas');
    await userEvent.keyboard('{Escape}');
    await page.getByRole('textbox', { name: 'Property field' }).click();
    await page.getByRole('button', { name: 'Shortcuts', exact: true }).click();
    await expect.poll(() => page.getByRole('dialog').element().textContent).toContain('Workbench');
    expect(page.getByRole('dialog').element().textContent).not.toContain('On Canvas');
  });

  it('runs configured global shortcuts from the compact chip and its popover without activating the chip for modified Enter', async () => {
    const guide = page.getByRole('button', { name: 'Shortcuts', exact: true });
    await act(() => (guide.element() as HTMLElement).focus());
    await userEvent.keyboard('{Control>}{Shift>}j{/Shift}{/Control}');
    await expect.poll(() => executed).toEqual(['app.invoke']);
    await userEvent.keyboard('{Control>}{Enter}{/Control}');
    expect(page.getByRole('dialog').query()).toBeNull();
    await userEvent.keyboard('{Enter}');
    await expect.element(page.getByRole('dialog')).toBeVisible();
    await userEvent.keyboard('{Control>}{Shift>}x{/Shift}{/Control}');
    await userEvent.keyboard('{Control>}j{/Control}');
    await expect.poll(() => executed).toEqual(['app.invoke', 'app.cancelQueueItem', 'app.openCommandPalette']);
  });
});
