/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { RegisteredWidget, WidgetViewProps } from '@workbench/widgetContracts';
import type * as WidgetRegistry from '@workbench/widgetRegistry';
import type * as WorkbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, useSensor, useSensors } from '@dnd-kit/core';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { createInstance } from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, type ReactNode, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { PrimaryMouseSensor } from './holdToDragSensor';
import { StatusBar } from './StatusBar';

const runtime = vi.hoisted(() => ({
  store: null as unknown as WorkbenchInternalStore,
  widgets: [] as RegisteredWidget[],
}));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const extensions = createExtensionRegistry();
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContext>()),
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchReload: () => () => Promise.resolve(),
  useWorkbenchExtensions: () => extensions,
  useOptionalWorkbenchExtensions: () => extensions,
  useWorkbenchCommands: () => runtime.store.commands,
  useOptionalWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchQueries: () => runtime.store.queries,
  useOptionalWorkbenchQueries: () => runtime.store.queries,
  useActiveProjectId: () =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => s.activeProject.id),
  useActiveProjectSelector: <T,>(select: (project: Project) => T, equal?: (a: T, b: T) => boolean) =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => select(s.activeProject), equal),
}));
vi.mock('@workbench/widgetRegistry', async (importOriginal) => ({
  ...(await importOriginal<typeof WidgetRegistry>()),
  getWidgetById: (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id),
  getWidgetsForRegion: (region: WidgetRegion) =>
    runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region)),
}));
const { registerWidgets } = await import('@workbench/widgetRegistry');
const getWidgetById = (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id);
const getWidgetsForRegion = (region: WidgetRegion) =>
  runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region));
const i18n = createInstance();
await i18n
  .use(initReactI18next)
  .init({ lng: 'en', resources: { en: { translation: await fetch('/locales/en.json').then((r) => r.json()) } } });

const DROP_STATE = { helperText: '', isActive: false, isAllowed: false };
const STATUS_MENU_LABEL = 'Status bar widgets';

/** A popover body with plain content and a row that owns its own context menu, as notification rows do. */
const FixtureView = ({ presentation }: WidgetViewProps) => {
  const [ownMenuOpen, setOwnMenuOpen] = useState(false);
  if (presentation === 'compact') {
    return <span>Fixture chip</span>;
  }
  return (
    <div>
      <p>Fixture body</p>
      <div
        data-own-menu=""
        onContextMenu={(event) => {
          event.preventDefault();
          setOwnMenuOpen(true);
        }}
      >
        {ownMenuOpen ? 'Row menu open' : 'Fixture row'}
      </div>
    </div>
  );
};

const makeState = (): WorkbenchState => {
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  project.widgetInstances = {
    fixture: {
      id: 'fixture',
      typeId: 'fixture',
      createdAt: '2026-11-14',
      state: { id: 'fixture', label: 'fixture', values: {}, version: 1 },
    },
  };
  for (const region of Object.values(project.widgetRegions)) {
    region.instanceIds = [];
    region.activeInstanceId = '';
  }
  project.widgetRegions.bottom.instanceIds = ['fixture'];
  return state;
};

/** Every open menu naming the status bar's widgets: the widgets menu, opened from the bar. */
const statusMenus = () =>
  [...document.querySelectorAll<HTMLElement>('[role="menu"]')].filter(
    (menu) => menu.textContent?.includes(STATUS_MENU_LABEL) && menu.closest('[data-state="open"]') !== null
  );

/** Menus open within a frame of their event; give a leaked one two frames to show before checking. */
const expectNoStatusMenu = async () => {
  for (let frame = 0; frame < 2; frame++) {
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  }
  expect(statusMenus()).toHaveLength(0);
};

/** Records whether the page's handlers left a right-click's default (the browser's own menu) in place. */
const lastContextMenuDefault = () => {
  const seen: { prevented: boolean | null } = { prevented: null };
  const record = (event: Event) => {
    seen.prevented = event.defaultPrevented;
  };
  window.addEventListener('contextmenu', record);
  return {
    seen,
    release: () => window.removeEventListener('contextmenu', record),
  };
};

/** The shell's mouse sensor, so a click on a chip activates it instead of starting a drag. */
const ShellDnd = ({ children }: { children: ReactNode }) => {
  const sensors = useSensors(useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }));
  return <DndContext sensors={sensors}>{children}</DndContext>;
};

let host: HTMLDivElement;
let root: Root;

beforeEach(async () => {
  await page.viewport(1280, 900);
  accountLifecycle.activate('status-bar-context-menu', ':user:status-bar-context-menu');
  runtime.widgets = registerWidgets([
    {
      id: 'fixture',
      label: 'Fixture',
      version: 1,
      icon: CircleIcon,
      allowedRegions: ['bottom'],
      allowMultiple: false,
      bottomPanel: 'popover',
      load: () => Promise.resolve({ view: FixtureView }),
      failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    },
  ]);
  runtime.store = createWorkbenchStore(makeState());
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;justify-content:flex-end;height:600px;width:1200px';
  document.body.append(host);
  root = createRoot(host);
  const focus = createTestFocusController({
    getProjectId: () => runtime.store.getSnapshot().activeProject.id,
    isFloating: () => false,
  });
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={focus}>
            <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
              <ShellDnd>
                <StatusBar dropState={DROP_STATE} />
              </ShellDnd>
            </WorkbenchWidgetRegistryProvider>
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  await expect.poll(() => host.textContent).toContain('Fixture chip');
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

describe('status bar context menus', () => {
  it('leaves a right-click inside a widget popover to the popover', async () => {
    await userEvent.click(page.getByText('Fixture chip'));
    const popup = page.getByRole('dialog');
    await expect.element(popup.getByText('Fixture body')).toBeVisible();
    const contextMenu = lastContextMenuDefault();

    try {
      await userEvent.click(popup.getByText('Fixture body'), { button: 'right' });
      // The browser's own menu applies: nothing on the page claimed the event.
      expect(contextMenu.seen.prevented).toBe(false);
      await expectNoStatusMenu();

      await userEvent.click(popup.getByText('Fixture row'), { button: 'right' });
      await expect.element(popup.getByText('Row menu open')).toBeVisible();
      await expectNoStatusMenu();
    } finally {
      contextMenu.release();
    }

    await expect.element(popup).toBeVisible();
  });

  it("opens the widgets menu for a right-click on the bar's own surface", async () => {
    const bar = host.querySelector<HTMLElement>('footer')!;
    const bounds = bar.getBoundingClientRect();

    // The middle of the bar is its spacer, between the start cluster and the end cluster.
    await userEvent.click(bar, { button: 'right', position: { x: bounds.width / 2, y: bounds.height / 2 } });

    await expect.poll(() => statusMenus()).toHaveLength(1);
  });
});
