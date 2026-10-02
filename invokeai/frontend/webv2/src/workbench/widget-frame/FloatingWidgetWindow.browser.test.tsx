/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FloatingWidgetState } from '@workbench/layoutContracts';
import type {
  RegisteredWidget,
  WidgetImplementation,
  WidgetManifest,
  WidgetViewProps,
} from '@workbench/widgetContracts';

import { ChakraProvider, HStack } from '@chakra-ui/react';
import { system } from '@theme/system';
import { closeWorkbenchSettings, settingsDialogStore } from '@workbench/settings/settingsDialogStore';
import i18next from 'i18next';
import { MapIcon, TagsIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/**
 * Floating windows must retain widget actions and settings while substituting their own shade/maximize/dock
 * controls.
 */

const windowMocks = vi.hoisted(() => ({
  actionsRegion: null as string | null,
  dockFloating: vi.fn(),
  useFailingWidget: false,
  focusFloating: vi.fn(),
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
  widgetRegions: { center: { activeInstanceId: null, instanceIds: [] } },
};

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: unknown) => unknown) => selector(project),
  useActiveProjectId: () => project.id,
  useWorkbenchQueries: () => ({
    getProject: (projectId: string) => (projectId === project.id ? project : null),
    isActiveProject: (projectId: string) => projectId === project.id,
  }),
  useWorkbenchCommands: () => ({ widgets: windowMocks }),
}));

vi.mock('@workbench/WorkbenchWidgetRegistryContext', () => ({
  useWorkbenchWidgetRegistry: () => ({
    getWidgetById: () => (windowMocks.useFailingWidget ? failingWidget : registeredWidget),
    getWidgetsForRegion: () => [],
  }),
}));

// The runtime needs the workbench store; the window's chrome is what is under
// test, and neither the stub view nor the stub actions touch the runtime.
vi.mock('./createWidgetRuntime', () => ({ useWidgetRuntime: () => ({}) }));

import { FloatingWidgetWindow } from './FloatingWidgetWindow';

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
      </HStack>
    );
  },
  headerMenu: () => <div data-testid="header-menu" />,
  view: () => <div data-testid="map-body" />,
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
          floating: { dock: 'Dock to panel', maximize: 'Maximize', move: 'Move {{label}} window', shade: 'Shade' },
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

const renderWindow = async (floatingState: FloatingWidgetState = state) => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FloatingWidgetWindow instanceId="image-map-instance" stackRank={0} state={floatingState} />
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

beforeEach(() => {
  closeWorkbenchSettings();
  windowMocks.actionsRegion = null;
  windowMocks.useFailingWidget = false;
  windowMocks.dockFloating.mockClear();
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
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to panel"]')).not.toBeNull();
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
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to panel"]')).not.toBeNull();
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

  it('keeps the widget actions reachable while the window is shaded', async () => {
    await renderWindow({ ...state, mode: 'shaded' });

    expect(host?.querySelector('button[aria-label="Toggle cluster labels"]')).not.toBeNull();
    expect(host?.querySelector('button[aria-label="Image Map settings"]')).not.toBeNull();
    expect(host?.querySelector('[data-testid="map-body"]')).toBeNull();
  });

  it('still docks from the title bar with the widget actions alongside', async () => {
    await renderWindow();

    await act(async () => {
      host?.querySelector<HTMLButtonElement>('button[aria-label="Dock to panel"]')?.click();
      await Promise.resolve();
    });

    expect(windowMocks.dockFloating).toHaveBeenCalledWith('image-map-instance');
  });
});

describe('FloatingWidgetWindow gestures', () => {
  const pointer = (type: string, clientX: number, clientY: number, buttons: number) =>
    new PointerEvent(type, { bubbles: true, buttons, clientX, clientY, pointerId: 1 });
  const nextFrame = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );
  const windowElement = () => host!.querySelector<HTMLElement>('[data-hotkey-widget-region="floating"]')!;

  it('renders a corner resize live and commits it once on release', async () => {
    await renderWindow();
    const corner = host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!;

    await act(() => corner.dispatchEvent(pointer('pointerdown', 0, 0, 1)));
    await act(() => window.dispatchEvent(pointer('pointermove', 60, 40, 1)));
    await nextFrame();

    expect(windowElement().style.width).toBe('560px');
    expect(windowElement().style.height).toBe('440px');
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();

    await act(() => window.dispatchEvent(pointer('pointerup', 60, 40, 0)));
    expect(windowMocks.setFloatingGeometry).toHaveBeenCalledOnce();

    await nextFrame();
    expect(windowElement().style.width).toBe('');
  });

  it('drops the live size when the window maximizes mid-resize', async () => {
    await renderWindow();
    const corner = host!.querySelector<HTMLElement>('[role="separator"][aria-valuemin]')!;

    await act(() => corner.dispatchEvent(pointer('pointerdown', 0, 0, 1)));
    await act(() => window.dispatchEvent(pointer('pointermove', 60, 40, 1)));
    await nextFrame();
    expect(windowElement().style.width).toBe('560px');

    await renderWindow({ ...state, mode: 'maximized' });
    await act(() => window.dispatchEvent(pointer('pointerup', 60, 40, 0)));

    expect(windowElement().style.width).toBe('');
    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
  });

  it('moves by the title bar and abandons the move on Escape', async () => {
    await renderWindow();
    const titleBar = host!.querySelector<HTMLElement>('[aria-label="Move Image Map window"]')!;

    await act(() => titleBar.dispatchEvent(pointer('pointerdown', 0, 0, 1)));
    await act(() => window.dispatchEvent(pointer('pointermove', 30, 20, 1)));
    await nextFrame();
    expect(windowElement().style.left).toContain('70px');

    await act(() => window.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    await act(() => window.dispatchEvent(pointer('pointerup', 30, 20, 0)));
    await nextFrame();

    expect(windowMocks.setFloatingGeometry).not.toHaveBeenCalled();
    expect(windowElement().style.left).toBe('');
  });
});
