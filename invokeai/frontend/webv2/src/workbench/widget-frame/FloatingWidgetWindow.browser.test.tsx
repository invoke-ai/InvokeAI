/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as draftRegistry from '@platform/react/draftRegistry';
import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type * as settingsStore from '@workbench/settings/store';
import type {
  RegisteredWidget,
  WidgetContributionSource,
  WidgetImplementation,
  WidgetManifest,
  WidgetViewProps,
} from '@workbench/widgetContracts';

import { ChakraProvider, HStack } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createExtensionRegistry, type ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider, type WorkbenchFocusController } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { closeWorkbenchSettings, settingsDialogStore } from '@workbench/settings/settingsDialogStore';
import i18next from 'i18next';
import { MapIcon, TagsIcon } from 'lucide-react';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/**
 * Floating windows must retain widget actions and settings while substituting their own shade/maximize/dock
 * controls.
 */

const windowMocks = vi.hoisted(() => ({
  actionsRegion: null as string | null,
  extensions: null as unknown as ExtensionRegistry,
  dockFloating: vi.fn(),
  flushWorkbenchDrafts: vi.fn(),
  layout: { setRegionCollapsed: vi.fn(), setRegionSize: vi.fn() },
  showFocusHighlight: true,
  useFailingWidget: false,
  useMissingWidget: false,
  useWideActions: false,
  raiseFloating: vi.fn(),
  setFloatingGeometry: vi.fn(),
  setFloatingMode: vi.fn(),
}));

const project = {
  id: 'project-1',
  widgetInstances: {
    'image-map-instance': {
      createdAt: 0,
      id: 'image-map-instance',
      state: { values: {} },
      typeId: 'image-map',
    },
  },
  widgetRegions: {
    center: { activeInstanceId: null, instanceIds: [] },
    right: { activeInstanceId: null, instanceIds: [], isCollapsed: false, sizePx: 320 },
  },
};

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: unknown) => unknown) => selector(project),
  useActiveProjectId: () => project.id,
  useWorkbenchQueries: () => ({
    getProject: (projectId: string) => (projectId === project.id ? project : null),
    isActiveProject: (projectId: string) => projectId === project.id,
  }),
  useWorkbenchCommands: () => ({ layout: windowMocks.layout, widgets: windowMocks }),
  useOptionalWorkbenchExtensions: () => windowMocks.extensions,
  useWorkbenchExtensions: () => windowMocks.extensions,
}));
// They need the whole application; the test registers the one binding it needs.
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

vi.mock('@platform/react/draftRegistry', async (importOriginal) => ({
  ...(await importOriginal<typeof draftRegistry>()),
  flushWorkbenchDrafts: windowMocks.flushWorkbenchDrafts,
}));

vi.mock('@workbench/settings/store', async (importOriginal) => {
  const original = await importOriginal<typeof settingsStore>();

  return {
    ...original,
    // Only the outline preference is steered; every other preference reads through.
    useWorkbenchPreferenceSelector: <Selected,>(
      selector: (preferences: ReturnType<typeof settingsStore.getWorkbenchPreferences>) => Selected
    ) => selector({ ...original.getWorkbenchPreferences(), showFocusRegionHighlight: windowMocks.showFocusHighlight }),
  };
});

vi.mock('@workbench/WorkbenchWidgetRegistryContext', () => ({
  useWorkbenchWidgetRegistry: () => ({
    getWidgetById: () =>
      windowMocks.useMissingWidget ? undefined : windowMocks.useFailingWidget ? failingWidget : registeredWidget,
    getWidgetsForRegion: () => [],
  }),
}));

// The runtime needs the workbench store; the window's chrome is what is under
// test, and neither the stub view nor the stub actions touch the runtime.
vi.mock('./createWidgetRuntime', () => ({ useWidgetRuntime: () => ({}) }));

import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';

import { FloatingWidgetWindow } from './FloatingWidgetWindow';
import { MissingWidgetFrame } from './WidgetRenderer';

// Floating chrome includes settings; overflow belongs to the docked header cluster.
const manifest = {
  allowFloating: true,
  allowedRegions: ['center', 'left', 'right'],
  failurePolicy: { isolateRenderFailure: false, onRegistrationFailure: 'disable' },
  icon: MapIcon,
  id: 'image-map',
  label: () => 'Image Map',
  settings: { id: 'imageMap', label: 'Image Map', fields: [], load: () => Promise.resolve({ Field: () => null }) },
  version: 1,
} as unknown as WidgetManifest;

/** A body with local state, to tell a hidden body from a remounted one. */
const MapBody = () => {
  const [count, setCount] = useState(0);

  return (
    <button data-testid="map-body" type="button" onClick={() => setCount(count + 1)}>
      {count}
    </button>
  );
};

const implementation: WidgetImplementation = {
  headerActions: ({ region }: WidgetViewProps) => {
    windowMocks.actionsRegion = region;

    // Widgets group their actions in a flex row, as the real header actions
    // do; the gear the window adds beside that group has to stay on its line.
    return (
      <HStack gap="1">
        <button aria-label="Toggle cluster labels" type="button">
          <TagsIcon />
        </button>
        {/* More actions than a narrow title bar can hold. */}
        {windowMocks.useWideActions
          ? ['Fit', 'Reset', 'Legend', 'Density', 'Export'].map((name) => (
              <button key={name} style={{ flexShrink: 0, width: '72px' }} type="button">
                {name}
              </button>
            ))
          : null}
      </HStack>
    );
  },
  headerMenu: () => <div data-testid="header-menu" />,
  view: () => <MapBody />,
};

const implementationPromise = Promise.resolve(implementation);
const registeredWidget = {
  implementation: { load: () => implementationPromise, retry: () => {} },
  manifest,
  status: 'enabled',
} as unknown as RegisteredWidget;

const failedLoad = Promise.reject(new Error('chunk unavailable'));
// Nothing awaits this before the render that consumes it; an unhandled
// rejection would fail the run on its own.
failedLoad.catch(() => undefined);
const failingWidget = {
  implementation: { load: () => failedLoad, retry: () => {} },
  // The body isolates its own failure, so only the title bar's containment is
  // under test here.
  manifest: { ...manifest, failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' } },
  status: 'enabled',
} as unknown as RegisteredWidget;

const state: FloatingWidgetState = {
  heightPx: 400,
  mode: 'windowed',
  returnRegion: 'right',
  stackOrder: 1,
  widthPx: 500,
  x: 40,
  y: 40,
};

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          floating: {
            destinations: { right: 'right panel' },
            dockTo: 'Dock to {{destination}}',
            maximize: 'Maximize',
            move: 'Move {{label}} window',
            resize: 'Resize window',
            resizeValue: '{{width}} by {{height}} pixels',
            restore: 'Restore',
            shade: 'Shade',
            unshade: 'Unshade',
          },
          labels: { imageMap: 'Image Map' },
          settingsLabel: '{{label}} settings',
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let testController = createTestFocusController();

const renderWindow = async (
  floatingState: FloatingWidgetState = state,
  controller: WorkbenchFocusController = testController,
  showUnavailablePanel = false
) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={controller}>
            <WorkbenchHotkeyRuntime />
            {showUnavailablePanel ? (
              <MissingWidgetFrame instanceId="image-map-instance" label="Image Map" region="right" typeId="image-map" />
            ) : null}
            <FloatingWidgetWindow instanceId="image-map-instance" stackRank={0} state={floatingState} />
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
    await implementationPromise;
  });
  // Flush the suspended chrome slot once more before asserting resolved actions.
  await act(async () => {
    await Promise.resolve();
  });
};

beforeEach(async () => {
  // Room for the 500x400 test window at (40, 40) without the viewport capping or clamping it.
  await page.viewport(1200, 800);
  closeWorkbenchSettings();
  windowMocks.actionsRegion = null;
  windowMocks.useFailingWidget = false;
  windowMocks.useMissingWidget = false;
  windowMocks.useWideActions = false;
  windowMocks.showFocusHighlight = true;
  testController = createTestFocusController();
  windowMocks.extensions = createExtensionRegistry();
  windowMocks.flushWorkbenchDrafts.mockClear();
  windowMocks.setFloatingMode.mockClear();
  windowMocks.dockFloating.mockClear();
  windowMocks.raiseFloating.mockClear();
  windowMocks.setFloatingGeometry.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('FloatingWidgetWindow chrome', () => {
  it("mounts the widget's own header actions in the title bar, for the floating region", async () => {
    await renderWindow();

    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).not.toBeNull();
    expect(host?.querySelector('[data-testid="map-body"]')).not.toBeNull();
    expect(windowMocks.actionsRegion).toBe('floating');
  });

  it('keeps one shared settings gear without duplicating the window layout controls', async () => {
    await renderWindow();

    expect(host?.querySelectorAll('button[aria-label="Image Map settings"]')).toHaveLength(1);
    expect(host?.querySelector('button[aria-label*="actions"]')).toBeNull();
    expect(host?.querySelector('button[aria-label="Float Window"]')).toBeNull();
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')).not.toBeNull();
  });

  it('keeps the settings gear in the title-bar strip instead of beneath the widget actions', async () => {
    await renderWindow();

    const action = host!.querySelector<HTMLButtonElement>('button[aria-label="Toggle cluster labels"]')!;
    const gear = host!.querySelector<HTMLButtonElement>('button[aria-label="Image Map settings"]')!;
    const shade = host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!;
    const [actionRect, gearRect, shadeRect] = [action, gear, shade].map((button) => button.getBoundingClientRect());

    expect(gearRect.top).toBeLessThan(actionRect.bottom);
    expect(gearRect.bottom).toBeGreaterThan(actionRect.top);
    // Strip order: widget actions, then the gear, then the window's controls.
    expect(gearRect.left).toBeGreaterThanOrEqual(actionRect.right);
    expect(shadeRect.left).toBeGreaterThanOrEqual(gearRect.right);
  });

  it('opens settings for the floating widget instance and remembers its gear for focus restoration', async () => {
    await renderWindow();
    const gear = host!.querySelector<HTMLButtonElement>('button[aria-label="Image Map settings"]')!;
    await act(async () => {
      gear.click();
      await Promise.resolve();
    });
    expect(settingsDialogStore.getSnapshot()).toMatchObject({
      isOpen: true,
      sectionId: 'imageMap',
      target: { instanceId: 'image-map-instance', projectId: 'project-1' },
    });
    expect(settingsDialogStore.getSnapshot().returnFocus).toBe(gear);
  });

  it('keeps the window usable when the widget implementation fails to load', async () => {
    windowMocks.useFailingWidget = true;

    await renderWindow();

    // Contain repeated rejected-resource throws in the title bar so the window's dock control survives.
    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).toBeNull();
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')).not.toBeNull();
  });

  it('does not move the window when an arrow key is pressed on a widget action', async () => {
    await renderWindow();

    const toggle = host?.querySelector<HTMLButtonElement>('button[aria-label="Toggle cluster labels"]');

    await act(async () => {
      toggle?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
      await Promise.resolve();
    });

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('still moves the window when an arrow key is pressed on the title bar itself', async () => {
    await renderWindow();

    const titleBar = host?.querySelector<HTMLElement>('[aria-label="Move Image Map window"]');

    await act(async () => {
      titleBar?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
      await Promise.resolve();
    });

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalled();
  });

  it('moves and resizes from the keyboard without the arrow keys also reaching the widget', async () => {
    // A widget-scoped arrow binding, as Preview and the gallery register theirs.
    const source: WidgetContributionSource = {
      instanceId: 'image-map-instance',
      projectId: 'project-1',
      region: 'floating',
      typeId: 'image-map',
    };
    const step = vi.fn();
    windowMocks.extensions.commands.register({ handler: step, id: 'image-map.step', source, title: 'Step' });
    windowMocks.extensions.hotkeys.register({
      commandId: 'image-map.step',
      defaultKeys: ['arrowright'],
      id: 'image-map.step',
      scope: 'widget',
      source,
      title: 'Step',
    });
    await renderWindow();
    const press = async (target: HTMLElement) => {
      await act(async () => {
        target.focus();
        await userEvent.keyboard('{ArrowRight}');
      });
    };

    // Inside the widget the binding is live, so a key the frame let through would reach it.
    await press(host!.querySelector<HTMLElement>('[data-testid="map-body"]')!);
    expect(step).toHaveBeenCalledTimes(1);

    await press(host!.querySelector<HTMLElement>('[aria-label="Move Image Map window"]')!);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(1);

    await press(host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(2);
    expect(step).toHaveBeenCalledTimes(1);
  });

  it('keeps the widget actions reachable while the window is shaded', async () => {
    await renderWindow({ ...state, mode: 'shaded' });

    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).not.toBeNull();
    expect(host?.querySelector('button[aria-label="Image Map settings"]')).not.toBeNull();
    expect(host!.querySelector('[data-testid="map-body"]')!.getClientRects()).toHaveLength(0);
  });

  it('hides a shaded body without unmounting it, saving drafts first', async () => {
    await renderWindow();
    const body = host!.querySelector<HTMLElement>('[data-testid="map-body"]')!;
    await act(() => body.click());
    expect(body.textContent).toBe('1');

    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.click());

    expect(windowMocks.flushWorkbenchDrafts).toHaveBeenCalledOnce();
    expect(windowMocks.setFloatingMode).toHaveBeenCalledWith('image-map-instance', 'shaded');
    expect(windowMocks.flushWorkbenchDrafts.mock.invocationCallOrder[0]).toBeLessThan(
      windowMocks.setFloatingMode.mock.invocationCallOrder[0]
    );

    await renderWindow({ ...state, mode: 'shaded' });
    // Rolled up to the title bar: the body is out of layout, and it is still the same element.
    expect(host!.querySelector('[data-floating-window]')!.getBoundingClientRect().height).toBeLessThan(60);
    expect(host!.querySelector('[data-testid="map-body"]')).toBe(body);
    expect(body.getClientRects()).toHaveLength(0);

    // Expanding saves nothing more, and the body comes back with its state.
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Unshade"]')!.click());
    expect(windowMocks.flushWorkbenchDrafts).toHaveBeenCalledOnce();
    await renderWindow();
    expect(host!.querySelector('[data-testid="map-body"]')?.textContent).toBe('1');
  });

  it('maximizes and restores on a title-bar double-click, but not from a title-bar button', async () => {
    await renderWindow();
    const titleBar = () => host!.querySelector<HTMLElement>('[aria-label="Move Image Map window"]')!;
    const doubleClick = (element: Element) =>
      act(() => element.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    await doubleClick(host!.querySelector('button[aria-label="Shade"]')!);
    await doubleClick(host!.querySelector('button[aria-label="Toggle cluster labels"]')!);
    expect(windowMocks.setFloatingMode).not.toHaveBeenCalled();

    await doubleClick(titleBar());
    expect(windowMocks.setFloatingMode).toHaveBeenLastCalledWith('image-map-instance', 'maximized');

    await renderWindow({ ...state, mode: 'maximized' });
    const rect = host!.querySelector('[data-floating-window]')!.getBoundingClientRect();
    expect([rect.left, rect.top, rect.width, rect.height]).toEqual([0, 0, 1200, 800]);

    await doubleClick(titleBar());
    // Restore goes back to the windowed geometry, which maximizing never touched.
    expect(windowMocks.setFloatingMode).toHaveBeenLastCalledWith('image-map-instance', 'windowed');
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('expands a collapsed window on a title-bar double-click instead of maximizing it', async () => {
    await renderWindow({ ...state, mode: 'shaded' });

    await act(() =>
      host!
        .querySelector('[aria-label="Move Image Map window"]')!
        .dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
    );

    expect(windowMocks.setFloatingMode).toHaveBeenCalledExactlyOnceWith('image-map-instance', 'windowed');
  });

  // The ring is a 2px outline offset by 2px.
  const FOCUS_RING_PX = 4;
  const showsWithRing = (action: Element): boolean => {
    const strip = host!.querySelector('[data-floating-actions]')!.getBoundingClientRect();
    const rect = action.getBoundingClientRect();

    return (
      rect.top - FOCUS_RING_PX >= strip.top &&
      rect.bottom + FOCUS_RING_PX <= strip.bottom &&
      rect.left - FOCUS_RING_PX >= strip.left &&
      rect.right + FOCUS_RING_PX <= strip.right
    );
  };

  it('leaves room for the focus ring of a contributed action inside the strip that holds them', async () => {
    await renderWindow();

    expect(showsWithRing(host!.querySelector('button[aria-label="Toggle cluster labels"]')!)).toBe(true);
  });

  it('brings a contributed action a narrow title bar cannot hold into view, ring included, when it takes focus', async () => {
    windowMocks.useWideActions = true;
    await renderWindow({ ...state, widthPx: 280 });

    const button = (name: string) =>
      [...host!.querySelectorAll<HTMLButtonElement>('[data-floating-actions] button')].find(
        (candidate) => (candidate.getAttribute('aria-label') ?? candidate.textContent) === name
      )!;

    // In Tab order: one in the middle of the strip, the last, then back to the first.
    for (const name of ['Legend', 'Image Map settings', 'Toggle cluster labels']) {
      expect(showsWithRing(button(name)), `${name} before focus`).toBe(false);

      await act(() => button(name).focus());

      expect(showsWithRing(button(name)), `${name} with focus`).toBe(true);
    }
  });

  it('keeps its own controls inside a minimum-width window whatever the widget contributes', async () => {
    windowMocks.useWideActions = true;
    await renderWindow({ ...state, widthPx: 280 });

    const frame = host!.querySelector('[data-floating-window]')!.getBoundingClientRect();
    const actions = host!.querySelector('[data-floating-actions]')!.getBoundingClientRect();
    const controls = host!.querySelector('[data-floating-controls]')!.getBoundingClientRect();

    expect(frame.width).toBe(280);
    for (const name of ['Shade', 'Maximize', 'Dock to right panel']) {
      const button = host!.querySelector(`button[aria-label="${name}"]`)!.getBoundingClientRect();

      expect(button.width).toBeGreaterThan(0);
      expect(button.left).toBeGreaterThanOrEqual(frame.left);
      expect(button.right).toBeLessThanOrEqual(frame.right);
    }
    // The contributed actions give way: they are clipped where the window's controls begin, and the title keeps
    // its icon rather than letting it slide over them.
    expect(actions.right).toBeLessThanOrEqual(controls.left);
    expect(
      host!.querySelector('[aria-label="Move Image Map window"] svg')!.getBoundingClientRect().right
    ).toBeLessThanOrEqual(actions.left);
  });

  it('still docks from the title bar with the widget actions alongside', async () => {
    await renderWindow();

    await act(async () => {
      host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')?.click();
      await Promise.resolve();
    });

    expect(windowMocks.dockFloating).toHaveBeenCalledWith('image-map-instance');
  });
});

describe('FloatingWidgetWindow focus', () => {
  const floatingWindow = () => host!.querySelector<HTMLElement>('[data-floating-window="image-map-instance"]')!;

  /** The frame that draws the window's border; the outline is its colour. */
  const borderColor = () => getComputedStyle(floatingWindow().firstElementChild!).borderTopColor;

  it('becomes the active, outlined window on pointer-down, and raises itself', async () => {
    const controller = createTestFocusController();
    await renderWindow(state, controller);
    const restingBorder = borderColor();

    await act(() => floatingWindow().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(controller.getTarget()).toEqual({ instanceId: 'image-map-instance', kind: 'floating' });
    expect(windowMocks.raiseFloating).toHaveBeenCalledWith('image-map-instance');
    // The outline is the border the user sees, not just a state attribute.
    await vi.waitFor(() => expect(borderColor()).not.toBe(restingBorder));

    controller.clear();
    await act(() => Promise.resolve());
    await vi.waitFor(() => expect(borderColor()).toBe(restingBorder));
  });

  it('becomes the active window and raises itself when keyboard focus arrives, with no press', async () => {
    const controller = createTestFocusController();
    await renderWindow(state, controller);

    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.focus());

    expect(controller.getTarget()).toEqual({ instanceId: 'image-map-instance', kind: 'floating' });
    expect(windowMocks.raiseFloating).toHaveBeenCalledWith('image-map-instance');
  });

  it('does not raise again when focus moves around inside the window that is already active', async () => {
    await renderWindow();
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.focus());
    expect(windowMocks.raiseFloating).toHaveBeenCalledOnce();

    // A Tab to the next control, or a closing dialog handing focus back: nothing to write, and nothing that
    // could pull this window back over one a recall has since revealed on top of it.
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Maximize"]')!.focus());
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.focus());

    expect(windowMocks.raiseFloating).toHaveBeenCalledOnce();
  });

  it('is active without an outline when the highlight preference is off', async () => {
    windowMocks.showFocusHighlight = false;
    const controller = createTestFocusController();
    await renderWindow(state, controller);
    const restingBorder = borderColor();

    await act(() => floatingWindow().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(controller.getTarget()).toEqual({ instanceId: 'image-map-instance', kind: 'floating' });
    expect(windowMocks.raiseFloating).toHaveBeenCalledWith('image-map-instance');
    expect(floatingWindow().getAttribute('data-highlighted')).toBe('false');
    expect(borderColor()).toBe(restingBorder);
  });

  it('takes keyboard focus from elsewhere when pressed, so keys follow the outline', async () => {
    const outside = document.createElement('button');
    document.body.append(outside);
    await renderWindow(state, createTestFocusController());
    outside.focus();

    await act(() =>
      host!.querySelector('[data-testid="map-body"]')!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    );

    expect(document.activeElement).toBe(floatingWindow());

    // Focus already inside the window stays where it is.
    const shade = host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!;
    shade.focus();
    await act(() =>
      host!.querySelector('[data-testid="map-body"]')!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    );

    expect(document.activeElement).toBe(shade);
    outside.remove();
  });

  it('is not activated or raised by hover', async () => {
    const controller = createTestFocusController();
    await renderWindow(state, controller);

    await act(() => {
      for (const type of ['pointerover', 'pointerenter', 'pointermove']) {
        floatingWindow().dispatchEvent(new PointerEvent(type, { bubbles: true }));
      }
      floatingWindow().dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
    });

    expect(controller.getTarget()).toBeNull();
    expect(windowMocks.raiseFloating).not.toHaveBeenCalled();
  });

  it('neither activates nor raises once its project has left the screen', async () => {
    // The window was rendered for project-1; the controller now answers for another project.
    const controller = createTestFocusController({ getProjectId: () => 'project-2' });
    await renderWindow(state, controller);

    await act(() => floatingWindow().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.focus());

    expect(controller.getTarget()).toBeNull();
    expect(windowMocks.raiseFloating).not.toHaveBeenCalled();
  });

  it('shows a ring on its frame when keyboard focus lands on the window itself', async () => {
    const before = document.createElement('button');
    host!.before(before);
    await renderWindow();
    const frame = floatingWindow().firstElementChild!;

    expect(getComputedStyle(frame).outlineStyle).toBe('none');

    // A keyboard user floats a panel: focus is moved onto the window root by script.
    before.focus();
    await userEvent.keyboard('{Tab}');
    await act(() => floatingWindow().focus());

    expect(floatingWindow().matches(':focus-visible')).toBe(true);
    // Drawn on the frame, which is its own layer and would cover a ring on the root.
    expect(getComputedStyle(frame)).toMatchObject({ outlineStyle: 'solid', outlineWidth: '2px' });
    expect(getComputedStyle(floatingWindow()).outlineStyle).toBe('none');
    before.remove();
  });

  it('can take scripted focus, and sends focus to its return region when it docks', async () => {
    const controller = createTestFocusController();
    const focusRegion = vi.spyOn(controller, 'focusRegion');
    await renderWindow(state, controller);

    await act(() => floatingWindow().focus());
    expect(document.activeElement).toBe(floatingWindow());

    await act(async () => {
      host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')?.click();
      await Promise.resolve();
    });

    expect(focusRegion).toHaveBeenCalledWith('right', 'image-map');
  });

  it('moves focus to the unavailable-widget panel when its floating window docks', async () => {
    windowMocks.useMissingWidget = true;
    const controller = createTestFocusController();
    // The panel the window docks into stands in for a widget whose view cannot render.
    await renderWindow(state, controller, true);
    const dock = host!.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')!;

    await act(() => dock.click());
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );

    expect(document.activeElement).toBe(host!.querySelector('[data-focus-region="right"]'));
  });
});

describe('FloatingWidgetWindow gestures', () => {
  const pointer = (type: string, clientX: number, clientY: number, buttons: number, init: PointerEventInit = {}) =>
    new PointerEvent(type, { bubbles: true, buttons, clientX, clientY, pointerId: 1, ...init });
  const nextFrame = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );
  const windowElement = () => host!.querySelector<HTMLElement>('[data-floating-window]')!;
  const titleBar = () => host!.querySelector<HTMLElement>('[aria-label="Move Image Map window"]')!;
  // The title bar's divider is a separator too; the resize grip is the one with a value.
  const corner = () => host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!;
  const edge = (name: string) => host!.querySelector<HTMLElement>(`[data-resize-edge="${name}"]`)!;
  /** What the gesture in flight is previewing, as written on the window element. */
  const preview = () => ({
    h: windowElement().style.getPropertyValue('--fw-preview-h'),
    mode: windowElement().dataset.floatingPreview,
    w: windowElement().style.getPropertyValue('--fw-preview-w'),
    x: windowElement().style.getPropertyValue('--fw-preview-x'),
    y: windowElement().style.getPropertyValue('--fw-preview-y'),
  });
  const NO_PREVIEW = { h: '', mode: undefined, w: '', x: '', y: '' };
  const rect = () => {
    const { height, left, top, width } = windowElement().getBoundingClientRect();

    return { height, left, top, width };
  };
  const press = (handle: HTMLElement, init?: PointerEventInit) =>
    act(() => handle.dispatchEvent(pointer('pointerdown', 100, 100, 1, init)));
  const moveBy = async (deltaX: number, deltaY: number, init?: PointerEventInit) => {
    await act(() => window.dispatchEvent(pointer('pointermove', 100 + deltaX, 100 + deltaY, 1, init)));
    await nextFrame();
  };
  const release = (deltaX: number, deltaY: number, init?: PointerEventInit) =>
    act(() => window.dispatchEvent(pointer('pointerup', 100 + deltaX, 100 + deltaY, 0, init)));
  /** Press, drag by an offset over several pointer moves, and (optionally) release. */
  const drag = async (handle: HTMLElement, deltaX: number, deltaY: number, { release: doRelease = true } = {}) => {
    await press(handle);
    for (const step of [0.25, 0.5, 0.75, 1]) {
      await moveBy(deltaX * step, deltaY * step);
    }
    if (doRelease) {
      await release(deltaX, deltaY);
    }
  };
  /** Every store command the window can issue, to show a drag in flight issues none. */
  const commandCalls = () =>
    [
      windowMocks.dockFloating,
      windowMocks.raiseFloating,
      windowMocks.setFloatingGeometry,
      windowMocks.setFloatingMode,
    ].reduce((total, command) => total + command.mock.calls.length, 0);

  it('renders a corner resize live, issues no command while the pointer moves, and commits once on release', async () => {
    await renderWindow();

    await press(corner());
    const commandsAfterPress = commandCalls();
    for (const step of [0.25, 0.5, 0.75, 1]) {
      await moveBy(60 * step, 40 * step);
    }

    expect(preview()).toEqual({ h: '440px', mode: 'windowed', w: '560px', x: '40px', y: '40px' });
    expect(rect()).toEqual({ height: 440, left: 40, top: 40, width: 560 });
    // Four coalesced pointer moves, zero store commands of any kind.
    expect(commandCalls()).toBe(commandsAfterPress);
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();

    await release(60, 40);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 440,
      widthPx: 560,
      x: 40,
      y: 40,
    });

    await nextFrame();
    expect(preview()).toEqual(NO_PREVIEW);
  });

  it.each([
    // The opposite side stays put: dragging the left edge left moves x and grows the width.
    ['w', -50, 30, 'ew-resize', { heightPx: 400, widthPx: 550, x: -10, y: 40 }],
    ['e', 50, 30, 'ew-resize', { heightPx: 400, widthPx: 550, x: 40, y: 40 }],
    ['s', 30, 50, 'ns-resize', { heightPx: 450, widthPx: 500, x: 40, y: 40 }],
    // The top edge stops at the viewport's top, so the title bar stays reachable.
    ['n', 30, -100, 'ns-resize', { heightPx: 440, widthPx: 500, x: 40, y: 0 }],
    ['nw', 20, 20, 'nwse-resize', { heightPx: 380, widthPx: 480, x: 60, y: 60 }],
    ['ne', 20, 20, 'nesw-resize', { heightPx: 380, widthPx: 520, x: 40, y: 60 }],
    // The minimum size holds, and the anchored right edge keeps the window from sliding.
    ['sw', 400, -400, 'nesw-resize', { heightPx: 200, widthPx: 280, x: 260, y: 40 }],
    // The band outside the bottom-right corner, around the labelled grip.
    ['se', 30, 20, 'nwse-resize', { heightPx: 420, widthPx: 530, x: 40, y: 40 }],
  ] as const)('resizes from the %s handle', async (name, deltaX, deltaY, cursor, expected) => {
    await renderWindow();

    expect(getComputedStyle(edge(name)).cursor).toBe(cursor);

    await drag(edge(name), deltaX, deltaY);

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', expected);
  });

  it('puts a handle under the pointer just outside and just inside every edge and corner, and no two at once', async () => {
    await renderWindow();
    const { bottom, left, right, top } = windowElement().getBoundingClientRect();
    const middleX = (left + right) / 2;
    const middleY = (top + bottom) / 2;
    const at = (x: number, y: number) => {
      const element = document.elementFromPoint(x, y);

      return element === corner() ? 'grip' : ((element as HTMLElement | null)?.dataset.resizeEdge ?? null);
    };

    // 2px outside the border, then 2px inside it.
    expect([at(middleX, top - 2), at(middleX, top + 2)]).toEqual(['n', 'n']);
    expect([at(middleX, bottom + 2), at(middleX, bottom - 2)]).toEqual(['s', 's']);
    expect([at(left - 2, middleY), at(left + 2, middleY)]).toEqual(['w', 'w']);
    expect([at(right + 2, middleY), at(right - 2, middleY)]).toEqual(['e', 'e']);
    expect([at(left - 2, top - 2), at(left + 2, top + 2)]).toEqual(['nw', 'nw']);
    expect([at(right + 2, top - 2), at(right - 2, top + 2)]).toEqual(['ne', 'ne']);
    expect([at(left - 2, bottom + 2), at(left + 2, bottom - 2)]).toEqual(['sw', 'sw']);
    // The bottom-right corner: the pointer-only band outside, the labelled grip inside.
    expect([at(right + 2, bottom + 2), at(right + 2, bottom - 6), at(right - 6, bottom + 2)]).toEqual([
      'se',
      'se',
      'se',
    ]);
    expect(at(right - 6, bottom - 6)).toBe('grip');
    // Past the reach on either side there is no handle: the content inside, nothing of the window outside.
    expect(at(middleX, top + 6)).toBeNull();
    expect(at(middleX, top - 6)).toBeNull();
  });

  it('leaves every title-bar control fully clickable beside the resize corners', async () => {
    await renderWindow();

    for (const name of ['Shade', 'Maximize', 'Dock to right panel', 'Toggle cluster labels']) {
      const button = host!.querySelector(`button[aria-label="${name}"]`)!;
      const { bottom, left, right, top } = button.getBoundingClientRect();

      // Just inside each side, at the middle: no resize handle may sit over any part of the control.
      for (const [x, y] of [
        [(left + right) / 2, top + 1],
        [(left + right) / 2, bottom - 1],
        [left + 1, (top + bottom) / 2],
        [right - 1, (top + bottom) / 2],
      ]) {
        expect(document.elementFromPoint(x, y)?.closest('button')).toBe(button);
      }
    }
  });

  it('keeps one labelled, keyboard-operable resize control and leaves the rest to the pointer', async () => {
    await renderWindow();

    const pointerOnly = [...host!.querySelectorAll<HTMLElement>('[data-resize-edge]')];

    expect(pointerOnly.map((handle) => handle.dataset.resizeEdge).sort()).toEqual([
      'e',
      'n',
      'ne',
      'nw',
      's',
      'se',
      'sw',
      'w',
    ]);
    for (const handle of pointerOnly) {
      expect(handle.getAttribute('aria-hidden')).toBe('true');
      expect(handle.hasAttribute('tabindex')).toBe(false);
      expect(handle.hasAttribute('role')).toBe(false);
    }
    expect(host!.querySelectorAll('[role="separator"][aria-valuemin]')).toHaveLength(1);
    expect(corner().tabIndex).toBe(0);
    expect(corner().getAttribute('aria-label')).toBe('Resize window');
    // A separator's value is one number; the size in words covers both dimensions.
    expect(corner().getAttribute('aria-valuetext')).toBe('500 by 400 pixels');
    expect(Number(corner().getAttribute('aria-valuemax'))).toBeGreaterThanOrEqual(500);
  });

  it('announces the size on screen through the resize control, following the viewport that caps it', async () => {
    await renderWindow({ ...state, heightPx: 700, widthPx: 900, x: 0, y: 0 });

    expect(corner().getAttribute('aria-valuenow')).toBe('900');
    expect(corner().getAttribute('aria-valuetext')).toBe('900 by 700 pixels');

    // The browser window shrinks under a window that nothing else re-renders.
    await act(() => page.viewport(600, 500));
    await nextFrame();

    expect(rect()).toMatchObject({ height: 500, width: 600 });
    expect(corner().getAttribute('aria-valuenow')).toBe('600');
    expect(corner().getAttribute('aria-valuemax')).toBe('600');
    expect(corner().getAttribute('aria-valuetext')).toBe('600 by 500 pixels');
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('never announces a minimum above the size a viewport narrower than it shows', async () => {
    await page.viewport(240, 500);
    await renderWindow({ ...state, x: 0, y: 0 });

    expect(corner().getAttribute('aria-valuenow')).toBe('240');
    expect(corner().getAttribute('aria-valuemin')).toBe('240');
  });

  it('has no resize handles while maximized or collapsed', async () => {
    for (const mode of ['maximized', 'shaded'] as const) {
      await renderWindow({ ...state, mode });

      expect(host!.querySelectorAll('[data-resize-edge]')).toHaveLength(0);
      expect(host!.querySelector('[role="separator"][aria-valuemin]')).toBeNull();
    }
  });

  it('resizes from the keyboard through the one labelled handle', async () => {
    await renderWindow();
    const key = (init: KeyboardEventInit) =>
      act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, ...init })));

    await key({ key: 'ArrowRight' });
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 400,
      widthPx: 516,
      x: 40,
      y: 40,
    });

    await key({ key: 'ArrowDown', shiftKey: true });
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 432,
      widthPx: 500,
      x: 40,
      y: 40,
    });

    await key({ key: 'Home' });
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 200,
      widthPx: 280,
      x: 40,
      y: 40,
    });
  });

  it('caps the displayed size and keeps a sliver reachable without rewriting what is stored', async () => {
    await page.viewport(600, 500);
    // Stored on a larger display: wider and taller than this viewport, and far off its right edge.
    await renderWindow({ ...state, heightPx: 900, widthPx: 900, x: 5000, y: 40 });

    expect(rect()).toEqual({ height: 500, left: 600 - 48, top: 40, width: 600 });

    // The viewport grows back: the stored rectangle returns, as far as it fits.
    await page.viewport(1200, 800);
    expect(rect()).toEqual({ height: 800, left: 1200 - 48, top: 40, width: 900 });
    // None of that wrote anything.
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('commits nothing for a press that never moves, even where the viewport clamps or caps the window', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 5000, y: 40 });

    // A click on the title bar — the first half of a double-click — and one on an edge, with pointer jitter that
    // nets to nothing.
    for (const handle of [titleBar(), edge('s')]) {
      await press(handle);
      await moveBy(3, 2);
      await moveBy(0, 0);
      await release(0, 0);
      await nextFrame();
    }

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(preview()).toEqual(NO_PREVIEW);
  });

  it('starts a move from where the window is on screen, and keeps the stored size', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 5000, y: 40 });

    await drag(titleBar(), -100, 20, { release: false });

    // No jump: the preview continues from the clamped position, 552px, not from the stored 5000px.
    expect(rect().left).toBe(452);

    await release(-100, 20);

    // The viewport's 600px cap on the width is not the user's choice to persist.
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 300,
      widthPx: 900,
      x: 452,
      y: 60,
    });
  });

  it('starts keyboard steps from where the window is on screen too', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 5000, y: 40 });

    await act(() => titleBar().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' })));
    // One 16px step left of the clamped 552px, keeping the stored size.
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 300,
      widthPx: 900,
      x: 536,
      y: 40,
    });

    await act(() =>
      titleBar().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown', shiftKey: true }))
    );
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 300,
      widthPx: 900,
      x: 552,
      y: 72,
    });
  });

  it('does not move a maximized window from the keyboard', async () => {
    await renderWindow({ ...state, mode: 'maximized' });

    await act(() => titleBar().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' })));

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('resizes the axis that was dragged from its size on screen, and leaves the other axis as stored', async () => {
    await page.viewport(600, 500);
    // 900px wide in storage, shown capped at 600px.
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 0, y: 40 });

    await drag(edge('s'), 0, 50);
    // Only the height was the user's to change; the 600px cap on the width is not persisted.
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 350,
      widthPx: 900,
      x: 0,
      y: 40,
    });

    await nextFrame();
    await drag(edge('e'), -50, 0);
    // Dragging the width starts from the 600px on screen, not the 900px stored for a larger display.
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 300,
      widthPx: 550,
      x: 0,
      y: 40,
    });

    await nextFrame();
    await act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 316,
      widthPx: 900,
      x: 0,
      y: 40,
    });
  });

  it('keeps the stored width when a viewport narrower than the minimum shows the window narrower still', async () => {
    await page.viewport(240, 500);
    // 520px wide in storage, shown capped at 240px: below the 280px minimum, which is not the user's choice either.
    await renderWindow({ ...state, heightPx: 300, widthPx: 520, x: 0, y: 40 });

    await drag(corner(), 0, 50);
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 350,
      widthPx: 520,
      x: 0,
      y: 40,
    });

    await nextFrame();
    await act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 316,
      widthPx: 520,
      x: 0,
      y: 40,
    });
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(2);
  });

  it('preserves an offscreen stored x position during a vertical resize', async () => {
    await page.viewport(240, 500);
    // The window is shown at x=192, but its stored x is still 1000 while the viewport is narrow.
    await renderWindow({ ...state, heightPx: 300, widthPx: 520, x: 1000, y: 40 });
    expect(rect().left).toBe(192);

    await act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 316,
      widthPx: 520,
      x: 1000,
      y: 40,
    });
  });

  it('preserves an offscreen stored y position during a horizontal resize', async () => {
    await page.viewport(500, 180);
    // The window is shown at y=132, but its stored y is still 800 while the viewport is short.
    await renderWindow({ ...state, heightPx: 300, widthPx: 400, x: 40, y: 800 });
    expect(rect().top).toBe(132);

    await act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 300,
      widthPx: 416,
      x: 40,
      y: 800,
    });
  });

  it('keeps the stored position of the axis a pointer resize leaves alone, wherever the viewport holds it', async () => {
    await page.viewport(500, 260);
    // Stored far off the bottom-right; shown at (452, 212) and capped to 260px tall.
    await renderWindow({ ...state, heightPx: 300, widthPx: 400, x: 1000, y: 800 });

    await drag(edge('e'), -50, 0);
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 300,
      widthPx: 350,
      x: 452,
      y: 800,
    });

    await nextFrame();
    await drag(edge('s'), 0, -40);
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 220,
      widthPx: 400,
      x: 1000,
      y: 212,
    });
  });

  it('does not grow a window past the viewport, so an edge at the cap resizes nothing instead of sliding it', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 300, widthPx: 600, x: 0, y: 40 });

    await drag(edge('w'), -80, 0, { release: false });

    // Already as wide as it can be shown: the anchored right edge stays where it is.
    expect(rect()).toMatchObject({ left: 0, width: 600 });

    await release(-80, 0);
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('commits nothing for a drag or key step against the viewport cap, so the cap is never what gets stored', async () => {
    await page.viewport(600, 500);
    // 900px wide in storage, shown capped at 600px.
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 0, y: 40 });

    await drag(edge('e'), 80, 0);
    await act(() => corner().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('moves a collapsed window by its title bar and keeps its stored height', async () => {
    await renderWindow({ ...state, mode: 'shaded' });

    await drag(titleBar(), 30, 20);

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 400,
      widthPx: 500,
      x: 70,
      y: 60,
    });
  });

  it('ignores a press that is not the primary button', async () => {
    await renderWindow();

    for (const handle of [titleBar(), edge('e'), corner()]) {
      await press(handle, { button: 2 });
      await moveBy(40, 40);
      await release(40, 40);
    }

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(preview()).toEqual(NO_PREVIEW);
  });

  it.each([
    ['the corner grip', () => corner()],
    ['an edge handle', () => edge('e')],
    ['the title bar', () => titleBar()],
  ] as const)('stands a gesture from %s down when the window maximizes under it', async (_name, handle) => {
    await renderWindow();

    await drag(handle(), 30, 20, { release: false });
    expect(preview().mode).toBe('windowed');

    await renderWindow({ ...state, mode: 'maximized' });
    // The new frame shows at once: the preview is for a mode the window is no longer in.
    expect(rect()).toEqual({ height: 800, left: 0, top: 0, width: 1200 });

    // The next the gesture hears from the pointer, it ends itself: no preview, no drag cursor, no commit.
    await moveBy(60, 60);
    expect(preview()).toEqual(NO_PREVIEW);
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);

    await release(60, 60);
    await nextFrame();
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('stands a title-bar move down when the window collapses under it', async () => {
    await renderWindow();

    await drag(titleBar(), 30, 20, { release: false });
    expect(preview().x).toBe('70px');

    await renderWindow({ ...state, mode: 'shaded' });
    // Back at its stored position, rolled up, before the pointer moves again.
    expect(rect()).toMatchObject({ left: 40, top: 40, width: 500 });
    expect(rect().height).toBeLessThan(60);

    await release(30, 20);
    await nextFrame();
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(preview()).toEqual(NO_PREVIEW);
  });

  it('moves by the title bar and abandons the move on Escape', async () => {
    await renderWindow();

    await drag(titleBar(), 30, 20, { release: false });
    expect(preview().x).toBe('70px');

    await act(() => window.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    await release(30, 20);
    await nextFrame();

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(preview()).toEqual(NO_PREVIEW);
  });

  it('keeps what was dragged so far when the pointer is taken away', async () => {
    await renderWindow();

    // The browser cancels the pointer (a touch gesture taken over, a system interruption).
    await drag(titleBar(), 30, 20, { release: false });
    await act(() => window.dispatchEvent(pointer('pointercancel', 130, 120, 0)));
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 400,
      widthPx: 500,
      x: 70,
      y: 60,
    });

    // The release happened outside the page; the next move arrives with no button held.
    await nextFrame();
    await drag(edge('e'), 40, 0, { release: false });
    await act(() => window.dispatchEvent(pointer('pointermove', 400, 400, 0)));
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 400,
      widthPx: 540,
      x: 40,
      y: 40,
    });
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(2);
  });

  it('lets a gesture that starts in the frame after a commit keep its own preview', async () => {
    await renderWindow();

    await drag(titleBar(), 30, 20);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledOnce();

    // No frame has passed: the first gesture's preview is still up, waiting to be cleared. The second gesture
    // starts from the stored rectangle (this test's store is a mock, so the commit did not move the window) and
    // the stale clear must not wipe it.
    await press(edge('e'));
    await act(() => window.dispatchEvent(pointer('pointermove', 150, 100, 1)));
    await nextFrame();
    await nextFrame();

    expect(preview()).toEqual({ h: '400px', mode: 'windowed', w: '550px', x: '40px', y: '40px' });

    await release(50, 0);
    expect(windowMocks.setFloatingGeometry).toHaveBeenLastCalledWith('image-map-instance', {
      heightPx: 400,
      widthPx: 550,
      x: 40,
      y: 40,
    });
  });

  it('hands the window to a second gesture: the first commits nothing and stops steering', async () => {
    await renderWindow();

    // One pointer is dragging the title bar when a second presses the grip.
    await drag(titleBar(), 30, 20, { release: false });
    await press(corner(), { pointerId: 2 });
    expect(preview()).toEqual(NO_PREVIEW);

    // The first pointer keeps moving and lets go: it no longer owns anything.
    await moveBy(200, 200);
    await release(200, 200);
    expect(preview()).toEqual(NO_PREVIEW);
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();

    await moveBy(60, 40, { pointerId: 2 });
    expect(preview()).toMatchObject({ h: '440px', w: '560px' });
    await release(60, 40, { pointerId: 2 });

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 440,
      widthPx: 560,
      x: 40,
      y: 40,
    });
  });

  it('maximizes on a real double-click without writing geometry for the two presses', async () => {
    await renderWindow();

    for (let click = 0; click < 2; click += 1) {
      await press(titleBar());
      await release(0, 0);
    }
    await act(() => titleBar().dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await nextFrame();

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(windowMocks.setFloatingMode).toHaveBeenCalledExactlyOnceWith('image-map-instance', 'maximized');
  });

  it('commits nothing and releases the pointer when the window unmounts mid-gesture', async () => {
    await renderWindow();

    await drag(titleBar(), 30, 20, { release: false });
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(true);

    // Docking, removing, or switching project all take the window away.
    await act(() => root?.render(null));
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);

    await act(() => window.dispatchEvent(pointer('pointermove', 300, 300, 1)));
    await release(200, 200);
    await nextFrame();

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });
});
