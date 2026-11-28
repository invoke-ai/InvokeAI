import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { registerModalPresence } from '@platform/ui/modalPresence';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFloatingWindowFocus, useFocusRegionProps } from '@workbench/focusRegions';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState, withArrangement } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const runtime = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveLayoutSelector: <Selected,>(selector: (layout: WorkbenchSnapshot['activeLayout']) => Selected) =>
    selector(useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot).activeLayout),
  useWorkbenchSelector: <Selected,>(selector: (snapshot: WorkbenchSnapshot) => Selected) =>
    selector(useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot)),
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
  const { activeLayout: project } = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
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
const layout = () => runtime.store.getSnapshot().activeLayout;
const regionElement = (region: WidgetRegion) => host.querySelector<HTMLElement>(`[data-focus-region="${region}"]`)!;
const press = (keys: string) => act(() => userEvent.keyboard(keys));
const regionChord = (arrow: string) =>
  IS_MAC_OS ? `{Control>}{Meta>}{${arrow}}{/Meta}{/Control}` : `{Alt>}{Shift>}{${arrow}}{/Shift}{/Alt}`;
const focusRegion = (region: WidgetRegion) => act(() => regionElement(region).focus());
const expectFocusedRegion = (region: WidgetRegion) =>
  vi.waitFor(() => {
    expect(document.activeElement?.closest('[data-focus-region]')).toBe(regionElement(region));
    expect(regionElement(region).dataset.highlighted).toBe('true');
  });

beforeEach(async () => {
  accountLifecycle.activate('workbench-navigation-test');
  await patchWorkbenchPreferences({ customHotkeys: {}, showFocusRegionHighlight: true });
  const state = withArrangement(
    createInitialWorkbenchState(),
    'compose',
    {
      bottom: ['gallery:bottom'],
      center: ['preview', 'canvas'],
      left: ['generate', 'generate-second'],
      right: ['gallery', 'layers'],
    },
    { 'generate-second': 'generate' }
  );
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

it('moves from the center before anything is focused, skips hidden regions, and stops at edges', async () => {
  await press(regionChord('ArrowLeft'));
  await expectFocusedRegion('left');
  regionElement('center').style.display = 'none';
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('right');
});

it('enters the center when the focused region has left the screen', async () => {
  await focusRegion('left');
  await expectFocusedRegion('left');
  regionElement('left').style.display = 'none';
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('center');
});

it('switches instances of the same type from a text field, wraps, and transfers actual focus', async () => {
  const textarea = host.querySelector<HTMLTextAreaElement>('[aria-label="generate text"]')!;
  await act(() => textarea.focus());
  await press('draft');
  await press('{Alt>}{PageDown}{/Alt}');
  expect(layout().widgetRegions.left.activeInstanceId).toBe('generate-second');
  await expectFocusedRegion('left');
  expect(getComputedStyle(document.activeElement!).outlineStyle).toBe('none');
  await press('{Tab}');
  expect(document.activeElement?.getAttribute('aria-label')).toBe('generate-second text');
  await press('{Alt>}{PageDown}{/Alt}');
  expect(layout().widgetRegions.left.activeInstanceId).toBe('generate');
  await expectFocusedRegion('left');
  expect(textarea.value).toBe('draft');
  await press('{Alt>}{PageUp}{/Alt}');
  expect(layout().widgetRegions.left.activeInstanceId).toBe('generate-second');
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
    expect(layout().widgetRegions.left.activeInstanceId).toBe('generate');
  } finally {
    release();
  }
  await act(() =>
    patchWorkbenchPreferences({
      customHotkeys: { 'app.focusRegionRight': ['alt+shift+n'], 'app.selectNextWidget': ['alt+shift+w'] },
    })
  );
  await press('{Alt>}{PageDown}{/Alt}');
  expect(layout().widgetRegions.left.activeInstanceId).toBe('generate');
  await press('{Alt>}{Shift>}w{/Shift}{/Alt}');
  expect(layout().widgetRegions.left.activeInstanceId).toBe('generate-second');
  await expectFocusedRegion('left');
  await press(regionChord('ArrowRight'));
  await expectFocusedRegion('left');
  await press('{Alt>}{Shift>}n{/Shift}{/Alt}');
  await expectFocusedRegion('center');
});

it('moves focus between the workflow graph and a collapsed workflow panel, preserving both views', async () => {
  await act(() => {
    runtime.store.commands.layout.applyPreset('automate');
    runtime.store.commands.widgets.patchValues('workflow', { editTab: 'json', panelMode: 'edit' });
    runtime.store.commands.layout.setRegionCollapsed('left', true);
  });
  const before = project();
  const beforeLayout = layout();
  expect(beforeLayout.widgetRegions.left.isCollapsed).toBe(true);
  await focusRegion('center');

  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('left');
  expect(layout().widgetRegions.left).toEqual({ ...beforeLayout.widgetRegions.left, isCollapsed: false });

  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('center');
  expect(layout().widgetRegions.center).toEqual(beforeLayout.widgetRegions.center);
  expect(project().widgetInstances).toBe(before.widgetInstances);
  expect(project().workflows).toBe(before.workflows);
});

it('shows the graph as the center’s guest where the workspace places only the form, writing no structure', async () => {
  await act(() => {
    runtime.store.commands.layout.applyPreset('compose');
    runtime.store.commands.widgets.open({ region: 'left', widgetId: 'workflow' });
  });
  const account = runtime.store.getSnapshot().account;
  await focusRegion('left');

  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('center');

  const front = layout().widgetRegions.center.activeInstanceId;
  expect(layout().guests).toEqual({ [front]: 'center' });
  expect(layout().widgetInstances[front]?.typeId).toBe('workflow');
  expect(runtime.store.getSnapshot().account).toBe(account);
});

it('shows the form as a guest window where the only workflow is the graph in the status bar panel', async () => {
  await act(() => {
    runtime.store.commands.layout.applyPreset('compose');
    runtime.store.commands.widgets.open({ region: 'bottom', widgetId: 'workflow' });
  });
  const beforeLayout = layout();
  await focusRegion('bottom');

  await press('{Shift>}Z{/Shift}');

  const [guestId] = Object.keys(layout().guests);
  expect(layout().widgetRegions).toBe(beforeLayout.widgetRegions);
  expect(layout().floatingWidgets?.[guestId!]).toMatchObject({ returnRegion: 'left' });
  expect(layout().widgetInstances[guestId!]?.typeId).toBe('workflow');
});

it('selects the existing Compose graph while preserving the shared left Workflow panel', async () => {
  await act(() => {
    runtime.store.commands.layout.applyPreset('compose');
    runtime.store.commands.widgets.open({ region: 'center', widgetId: 'workflow' });
  });
  const graphInstanceId = layout().widgetRegions.center.activeInstanceId;
  expect(project().widgetInstances[graphInstanceId]?.typeId).toBe('workflow');

  await act(() => {
    runtime.store.commands.widgets.open({ region: 'left', widgetId: 'workflow' });
    runtime.store.commands.widgets.select({ region: 'center', widgetId: 'preview' });
  });
  const beforeLayout = layout();
  expect(beforeLayout.widgetRegions.left.activeInstanceId).toBe(graphInstanceId);
  expect(beforeLayout.widgetRegions.center.activeInstanceId).toBe('preview');
  expect(beforeLayout.widgetRegions.center.instanceIds).toContain(graphInstanceId);
  await focusRegion('left');

  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('center');
  expect(layout().widgetRegions.left).toEqual(beforeLayout.widgetRegions.left);
  expect(layout().panels.isLeftOpen).toBe(true);
  expect(layout().widgetRegions.center).toEqual({
    ...beforeLayout.widgetRegions.center,
    activeInstanceId: graphInstanceId,
  });

  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('left');
  expect(layout().widgetRegions.left).toEqual(beforeLayout.widgetRegions.left);
  expect(layout().widgetRegions.center.activeInstanceId).toBe(graphInstanceId);
});

it('reopens a bottom workflow graph using the remapped shortcut only from workflow surfaces', async () => {
  await act(() => {
    runtime.store.commands.widgets.open({ region: 'bottom', widgetId: 'workflow' });
    runtime.store.commands.widgets.open({ region: 'left', widgetId: 'workflow' });
    runtime.store.commands.layout.setRegionCollapsed('bottom', true);
  });
  const before = project();
  const beforeLayout = layout();
  expect(beforeLayout.widgetRegions.bottom.isCollapsed).toBe(true);

  await focusRegion('right');
  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('right');
  expect(project()).toBe(before);

  const workflowId = beforeLayout.widgetRegions.left.activeInstanceId;
  const textarea = host.querySelector<HTMLTextAreaElement>(`[aria-label="${workflowId} text"]`)!;
  await act(() => textarea.focus());
  await press('{Shift>}Z{/Shift}');
  expect(textarea.value).toBe('Z');
  expect(document.activeElement).toBe(textarea);
  expect(project()).toBe(before);

  await focusRegion('left');
  await act(() => patchWorkbenchPreferences({ customHotkeys: { 'app.switchWorkflowView': ['shift+j'] } }));
  await press('{Shift>}Z{/Shift}');
  await expectFocusedRegion('left');
  expect(project()).toBe(before);

  await press('{Shift>}J{/Shift}');
  await expectFocusedRegion('bottom');
  expect(layout().widgetRegions.bottom).toEqual({ ...beforeLayout.widgetRegions.bottom, isCollapsed: false });
  expect(layout().widgetRegions.center).toEqual(beforeLayout.widgetRegions.center);
  await press('{Shift>}J{/Shift}');
  await expectFocusedRegion('left');
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
