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
import { createDeferredResource } from '@workbench/deferredResource';
import { createExtensionRegistry, type ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider, type WorkbenchFocusController } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { closeWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { createWidgetRecovery } from '@workbench/widget-frame/widgetRecoveryStore';
import { rearrangeWithoutCustomizeInEachTest } from '@workbench/workspace/structureLock.testing';
import i18next from 'i18next';
import { MapIcon, TagsIcon } from 'lucide-react';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const windowMocks = vi.hoisted(() => ({
  actionsRegion: null as string | null,
  closeFloating: vi.fn(),
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

const recoveryOwner = vi.hoisted(() => ({
  widgetRecovery: new Map(),
  getSnapshot: () => ({ activeProject: project }),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchInternalStore: () => recoveryOwner,
  useWorkbenchReload: () => async () => {},
  useActiveProjectSelector: (selector: (project: unknown) => unknown) => selector(project),
  useActiveProjectId: () => project.id,
  useWorkbenchQueries: () => ({
    getProject: (projectId: string) => (projectId === project.id ? project : null),
    getSnapshot: () => ({
      activeProject: { ...project, layout: { panels: { isBottomOpen: true, isLeftOpen: true, isRightOpen: true } } },
    }),
    isActiveProject: (projectId: string) => projectId === project.id,
  }),
  useWorkbenchCommands: () => ({ layout: windowMocks.layout, widgets: windowMocks }),
  useOptionalWorkbenchExtensions: () => windowMocks.extensions,
  useOptionalWorkbenchSelector: (_selector: unknown, fallback: unknown) => fallback,
  useWorkbenchExtensions: () => windowMocks.extensions,
}));
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

vi.mock('@platform/react/draftRegistry', async (importOriginal) => ({
  ...(await importOriginal<typeof draftRegistry>()),
  flushWorkbenchDrafts: windowMocks.flushWorkbenchDrafts,
}));

vi.mock('@workbench/settings/store', async (importOriginal) => {
  const original = await importOriginal<typeof settingsStore>();

  return {
    ...original,
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

vi.mock('./createWidgetRuntime', () => ({ useWidgetRuntime: () => ({}) }));

import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';

import { FloatingWidgetWindow } from './FloatingWidgetWindow';
import { MissingWidgetFrame } from './WidgetRenderer';

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

const implementationResource = createDeferredResource(() => Promise.resolve(implementation));
const implementationPromise = implementationResource.load();
const registeredWidget = {
  implementation: implementationResource,
  manifest,
  status: 'enabled',
} as unknown as RegisteredWidget;

const failingWidget = {
  implementation: createDeferredResource<WidgetImplementation>(() => Promise.reject(new Error('chunk unavailable'))),
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
            close: 'Close window',
            closeNamed: 'Close {{label}} window',
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
  showUnavailablePanel = false,
  showWindow = true
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
            {showWindow ? (
              <FloatingWidgetWindow instanceId="image-map-instance" stackRank={0} state={floatingState} />
            ) : null}
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
  windowMocks.closeFloating.mockClear();
  windowMocks.raiseFloating.mockClear();
  windowMocks.setFloatingGeometry.mockClear();
  recoveryOwner.widgetRecovery.set('image-map-instance', createWidgetRecovery());
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  recoveryOwner.widgetRecovery.clear();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

// Dock and Close are structure: these windows belong to an account that rearranges without Customize.
rearrangeWithoutCustomizeInEachTest();

describe('FloatingWidgetWindow chrome', () => {
  it("mounts the widget's own header actions in the title bar, for the floating region", async () => {
    await renderWindow();

    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).not.toBeNull();
    expect(host?.querySelector('[data-testid="map-body"]')).not.toBeNull();
    expect(windowMocks.actionsRegion).toBe('floating');
  });

  it('keeps the window usable when the widget implementation fails to load', async () => {
    windowMocks.useFailingWidget = true;

    await renderWindow();

    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).toBeNull();
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to right panel"]')).not.toBeNull();
  });

  it('moves and resizes from the keyboard without the arrow keys also reaching the widget', async () => {
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

    await press(host!.querySelector<HTMLElement>('[data-testid="map-body"]')!);
    expect(step).toHaveBeenCalledTimes(1);

    await press(host!.querySelector<HTMLElement>('[aria-label="Move Image Map window"]')!);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(1);

    await press(host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!);
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledTimes(2);
    expect(step).toHaveBeenCalledTimes(1);
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
    expect(host!.querySelector('[data-floating-window]')!.getBoundingClientRect().height).toBeLessThan(60);
    expect(host!.querySelector('[data-testid="map-body"]')).toBe(body);
    expect(body.getClientRects()).toHaveLength(0);

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
    expect(windowMocks.setFloatingMode).toHaveBeenLastCalledWith('image-map-instance', 'windowed');
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });
});

describe('FloatingWidgetWindow focus', () => {
  const floatingWindow = () => host!.querySelector<HTMLElement>('[data-floating-window="image-map-instance"]')!;

  const borderColor = () => getComputedStyle(floatingWindow().firstElementChild!).borderTopColor;

  it('becomes the active, outlined window on pointer-down, and raises itself', async () => {
    const controller = createTestFocusController();
    await renderWindow(state, controller);
    const restingBorder = borderColor();

    await act(() => floatingWindow().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(controller.getTarget()).toEqual({ instanceId: 'image-map-instance', kind: 'floating' });
    expect(windowMocks.raiseFloating).toHaveBeenCalledWith('image-map-instance');
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

  it('neither activates nor raises once its project has left the screen', async () => {
    const controller = createTestFocusController({ getProjectId: () => 'project-2' });
    await renderWindow(state, controller);

    await act(() => floatingWindow().dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    await act(() => host!.querySelector<HTMLButtonElement>('button[aria-label="Shade"]')!.focus());

    expect(controller.getTarget()).toEqual({ kind: 'region', region: 'center' });
    expect(windowMocks.raiseFloating).not.toHaveBeenCalled();
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
  const corner = () => host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!;
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
  const drag = async (handle: HTMLElement, deltaX: number, deltaY: number, { release: doRelease = true } = {}) => {
    await press(handle);
    for (const step of [0.25, 0.5, 0.75, 1]) {
      await moveBy(deltaX * step, deltaY * step);
    }
    if (doRelease) {
      await release(deltaX, deltaY);
    }
  };
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

  it('caps the displayed size and keeps a sliver reachable without rewriting what is stored', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 900, widthPx: 900, x: 5000, y: 40 });

    expect(rect()).toEqual({ height: 500, left: 600 - 48, top: 40, width: 600 });

    await page.viewport(1200, 800);
    expect(rect()).toEqual({ height: 800, left: 1200 - 48, top: 40, width: 900 });
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('starts a move from where the window is on screen, and keeps the stored size', async () => {
    await page.viewport(600, 500);
    await renderWindow({ ...state, heightPx: 300, widthPx: 900, x: 5000, y: 40 });

    await drag(titleBar(), -100, 20, { release: false });

    // No jump: the preview continues from the clamped position, 552px, not from the stored 5000px.
    expect(rect().left).toBe(452);

    await release(-100, 20);

    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledExactlyOnceWith('image-map-instance', {
      heightPx: 300,
      widthPx: 900,
      x: 452,
      y: 60,
    });
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

  it('commits nothing and releases the pointer when the window unmounts mid-gesture', async () => {
    await renderWindow();

    await drag(titleBar(), 30, 20, { release: false });
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(true);

    await act(() => root?.render(null));
    expect(document.documentElement.hasAttribute('data-pointer-drag')).toBe(false);

    await act(() => window.dispatchEvent(pointer('pointermove', 300, 300, 1)));
    await release(200, 200);
    await nextFrame();

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });
});
