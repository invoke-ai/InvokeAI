/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { registerModalPresence } from '@platform/ui/modalPresence';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFloatingWindowFocus, useFocusRegionProps } from '@workbench/focusRegions';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// Real command registrations, hotkey routing, preferences, focus, registry, and project reducer.
// Only provider plumbing, remote services, and the expensive widget views are replaced.
const runtime = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) => {
    const snapshot = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
    return selector(snapshot.activeProject);
  },
  useOptionalWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchQueries: () => runtime.store.queries,
  useWorkbenchSubscription: () => runtime.store.subscribe,
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
}));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  setClientStateValue: () => Promise.resolve(),
}));

import { IS_MAC_OS } from './keys';
import { WorkbenchHotkeyRuntime } from './WorkbenchHotkeyRuntime';

const positions = {
  left: { height: 400, left: 0, top: 0, width: 180 },
  center: { height: 400, left: 200, top: 0, width: 380 },
  right: { height: 400, left: 600, top: 0, width: 180 },
  bottom: { height: 120, left: 0, top: 420, width: 780 },
};

const Region = ({ region }: { region: WidgetRegion }) => {
  const { activeProject: project } = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
  const state = project.widgetRegions[region];
  return (
    <Box
      {...useFocusRegionProps(region)}
      style={{ ...positions[region], display: state.isCollapsed ? 'none' : undefined, position: 'absolute' }}
      tabIndex={-1}
    >
      {state.instanceIds.map((id) => (
        <div
          key={id}
          data-hotkey-widget-instance-id={id}
          data-hotkey-widget-region={region}
          data-hotkey-widget-type-id={project.widgetInstances[id]?.typeId}
          style={{ display: id === state.activeInstanceId ? 'block' : 'none' }}
        >
          <textarea aria-label={`${id} text`} />
          <button type="button">{id} control</button>
        </div>
      ))}
    </Box>
  );
};

const Floating = () => {
  const { activate } = useFloatingWindowFocus('image-map', runtime.store.getSnapshot().activeProject.id);
  return (
    <div
      data-floating-window="image-map"
      data-hotkey-widget-instance-id="image-map"
      data-hotkey-widget-region="floating"
      data-hotkey-widget-type-id="image-map"
      onFocusCapture={() => activate()}
      onPointerDownCapture={(event) => {
        activate({ byPointer: true });
        event.currentTarget.focus({ preventScroll: true });
      }}
      style={{ height: 180, left: 250, position: 'absolute', top: 80, width: 200 }}
      tabIndex={-1}
    />
  );
};

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const project = () => runtime.store.getSnapshot().activeProject;
const regionElement = (region: WidgetRegion) => host.querySelector<HTMLElement>(`[data-focus-region="${region}"]`)!;
const press = (keys: string) => act(() => userEvent.keyboard(keys));
/** The default region-focus chord: Control+Option+Arrow on macOS, Alt+Shift+Arrow elsewhere. */
const regionChord = (arrow: string) =>
  IS_MAC_OS ? `{Control>}{Alt>}{${arrow}}{/Alt}{/Control}` : `{Alt>}{Shift>}{${arrow}}{/Shift}{/Alt}`;
const focusRegion = (region: WidgetRegion) => act(() => regionElement(region).focus());
const expectFocusedRegion = (region: WidgetRegion) =>
  vi.waitFor(() => {
    expect(document.activeElement?.closest('[data-focus-region]')).toBe(regionElement(region));
    expect(regionElement(region).dataset.highlighted).toBe('true');
  });

beforeEach(async () => {
  accountLifecycle.activate('workbench-navigation-test');
  await patchWorkbenchPreferences({ customHotkeys: {}, showFocusRegionHighlight: true });
  const state = createInitialWorkbenchState();
  const initial = state.projects[0]!;
  initial.widgetInstances['generate-second'] = { ...initial.widgetInstances.generate!, id: 'generate-second' };
  const ids = {
    left: ['generate', 'generate-second'],
    center: ['preview', 'canvas'],
    right: ['gallery', 'layers'],
    bottom: ['queue'],
  };
  for (const region of Object.keys(ids) as WidgetRegion[]) {
    initial.widgetRegions[region] = {
      ...initial.widgetRegions[region],
      activeInstanceId: ids[region][0]!,
      instanceIds: ids[region],
      isCollapsed: false,
    };
  }
  initial.layout.panels = { isBottomOpen: true, isLeftOpen: true, isRightOpen: true };
  runtime.store = createWorkbenchStore(state);
  runtime.store.commands.widgets.open({ region: 'center', widgetId: 'image-map' });
  runtime.store.commands.widgets.float('image-map');
  runtime.extensions = createExtensionRegistry();
  queryClient = new QueryClient();
  host = document.createElement('div');
  host.style.position = 'relative';
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <WorkbenchFocusProvider>
            <WorkbenchHotkeyRuntime />
            <Region region="left" />
            <Region region="center" />
            <Region region="right" />
            <Region region="bottom" />
            <Floating />
          </WorkbenchFocusProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  host?.remove();
  accountLifecycle.invalidate();
});

it('moves between visible regions in all four directions, preserving widget selections', async () => {
  const before = project();
  await focusRegion('left');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('center');
  expect(getComputedStyle(regionElement('center')).outlineStyle).toBe('none');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
  await press(regionChord('ArrowLeft'));
  await expectFocusedRegion('center');
  await press(regionChord('ArrowDown'));
  await expectFocusedRegion('bottom');
  await press(regionChord('ArrowUp'));
  await expectFocusedRegion('center');
  expect(project()).toBe(before);
});

it('skips hidden regions, stops at edges, and enters the center without an initial target', async () => {
  await press(regionChord('ArrowLeft'));
  await expectFocusedRegion('center');
  regionElement('center').style.display = 'none';
  await focusRegion('left');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
});

it('switches instances of the same type from a text field, wraps, and transfers actual focus', async () => {
  const textarea = host.querySelector<HTMLTextAreaElement>('[aria-label="generate text"]')!;
  await act(() => textarea.focus());
  await press('draft');
  await press('{Alt>}{PageDown}{/Alt}');
  expect(project().widgetRegions.left.activeInstanceId).toBe('generate-second');
  await expectFocusedRegion('left');
  expect(getComputedStyle(document.activeElement!).outlineStyle).toBe('none');
  await press('{Tab}');
  expect(document.activeElement?.getAttribute('aria-label')).toBe('generate-second text');
  await press('{Alt>}{PageDown}{/Alt}');
  expect(project().widgetRegions.left.activeInstanceId).toBe('generate');
  await expectFocusedRegion('left');
  expect(textarea.value).toBe('draft');
  await press('{Alt>}{PageUp}{/Alt}');
  expect(project().widgetRegions.left.activeInstanceId).toBe('generate-second');
  await expectFocusedRegion('left');
});

it('leaves normal Tab and arrow editing intact and navigates out of editable fields', async () => {
  const textarea = host.querySelector<HTMLTextAreaElement>('[aria-label="generate text"]')!;
  await act(() => textarea.focus());
  await press('text{ArrowLeft}');
  expect(textarea.selectionStart).toBe(3);
  await press('{Tab}');
  expect(document.activeElement?.textContent).toBe('generate control');
  await press('{Shift>}{Tab}{/Shift}');
  expect(document.activeElement).toBe(textarea);
  expect(getComputedStyle(textarea).outlineStyle).not.toBe('none');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('center');
  await act(() => patchWorkbenchPreferences({ showFocusRegionHighlight: false }));
  expect(getComputedStyle(regionElement('center')).outlineStyle).not.toBe('none');
});

it('respects modal suspension and custom bindings', async () => {
  await focusRegion('left');
  const release = registerModalPresence();
  try {
    await press(regionChord('ArrowRight'));
    await press('{Alt>}{PageDown}{/Alt}');
    await expectFocusedRegion('left');
    expect(project().widgetRegions.left.activeInstanceId).toBe('generate');
  } finally {
    release();
  }
  await act(() =>
    patchWorkbenchPreferences({
      customHotkeys: { 'app.focusRegionRight': ['alt+shift+n'], 'app.selectNextWidget': ['alt+shift+w'] },
    })
  );
  await press('{Alt>}{PageDown}{/Alt}');
  expect(project().widgetRegions.left.activeInstanceId).toBe('generate');
  await press('{Alt>}{Shift>}w{/Shift}{/Alt}');
  expect(project().widgetRegions.left.activeInstanceId).toBe('generate-second');
  await expectFocusedRegion('left');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('left');
  await press('{Alt>}{Shift>}n{/Shift}{/Alt}');
  await expectFocusedRegion('center');
});

it('does not cycle a floating window and can move from it to a docked region', async () => {
  const window = host.querySelector<HTMLElement>('[data-floating-window]')!;
  await act(() => window.focus());
  const before = project();
  await press('{Alt>}{PageDown}{/Alt}');
  expect(project()).toBe(before);
  expect(document.activeElement).toBe(window);
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
});

it('can leave a floating window overlapping every visible docked region', async () => {
  const window = host.querySelector<HTMLElement>('[data-floating-window]')!;
  window.style.left = '150px';
  window.style.width = '520px';
  regionElement('bottom').style.display = 'none';
  await act(() => window.focus());
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
  await act(() => userEvent.click(window));
  // A pending dock-focus settle must not reclaim the window the user deliberately returned to.
  await act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );
  expect(document.activeElement).toBe(window);
  await press(regionChord('ArrowLeft'));
  await expectFocusedRegion('left');
});

it('returns to the underlying dock when there is no region beyond the floating window', async () => {
  const window = host.querySelector<HTMLElement>('[data-floating-window]')!;
  for (const region of ['left', 'right', 'bottom'] as const) {
    regionElement(region).style.display = 'none';
  }
  for (const key of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']) {
    await act(() => userEvent.click(window));
    expect(document.activeElement).toBe(window);
    await press(regionChord(key));
    await expectFocusedRegion('center');
  }
});
