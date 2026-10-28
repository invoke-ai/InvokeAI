/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// The hotkey module reads the platform once, as it loads: this file runs the real runtime under a macOS identity.
// tinykeys resolves `mod` from navigator.platform, so both platform reads must say macOS.
// Neither fake is restored: browser test files are isolated, and both readers consult the platform only at load.
vi.hoisted(() => {
  Object.defineProperty(navigator, 'userAgentData', { configurable: true, value: { platform: 'macOS' } });
  Object.defineProperty(navigator, 'platform', { configurable: true, value: 'MacIntel' });
});

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
};

const Region = ({ region }: { region: 'center' | 'left' }) => {
  const { activeProject: project } = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
  const id = project.widgetRegions[region].activeInstanceId;
  return (
    <Box {...useFocusRegionProps(region)} style={{ ...positions[region], position: 'absolute' }} tabIndex={-1}>
      <div
        data-hotkey-widget-instance-id={id}
        data-hotkey-widget-region={region}
        data-hotkey-widget-type-id={project.widgetInstances[id ?? '']?.typeId}
      >
        <textarea aria-label={`${region} text`} />
      </div>
    </Box>
  );
};

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const regionElement = (region: WidgetRegion) => host.querySelector<HTMLElement>(`[data-focus-region="${region}"]`)!;
const press = (keys: string) => act(() => userEvent.keyboard(keys));
const focusedRegion = () => document.activeElement?.closest('[data-focus-region]');

beforeEach(async () => {
  accountLifecycle.activate('workbench-navigation-macos-test');
  await patchWorkbenchPreferences({ customHotkeys: {} });
  const state = createInitialWorkbenchState();
  const initial = state.projects[0]!;
  initial.widgetRegions.left = {
    ...initial.widgetRegions.left,
    activeInstanceId: 'generate',
    instanceIds: ['generate'],
  };
  initial.widgetRegions.center = {
    ...initial.widgetRegions.center,
    activeInstanceId: 'canvas',
    instanceIds: ['canvas'],
  };
  initial.layout.panels = { isBottomOpen: false, isLeftOpen: true, isRightOpen: false };
  runtime.store = createWorkbenchStore(state);
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

it('leaves Option+Shift+Arrow to the text field and Control+Option+Arrow to VoiceOver, and moves between regions with Control+Command+Arrow', async () => {
  expect(IS_MAC_OS).toBe(true);
  const textarea = host.querySelector<HTMLTextAreaElement>('[aria-label="left text"]')!;
  const claimed: string[] = [];
  const observe = (event: KeyboardEvent) => {
    if (event.defaultPrevented) {
      claimed.push(
        `${event.altKey ? 'alt+' : ''}${event.ctrlKey ? 'ctrl+' : ''}${event.metaKey ? 'meta+' : ''}${event.key}`
      );
    }
  };
  window.addEventListener('keydown', observe);
  try {
    await act(() => textarea.focus());
    await press('two words');
    // Option+Shift+Left selects the word before the caret, as it does in every macOS text field.
    await press('{Alt>}{Shift>}{ArrowLeft}{/Shift}{/Alt}');
    // VoiceOver's navigation chord (VO+Arrow) must stay with the screen reader.
    await press('{Control>}{Alt>}{ArrowRight}{/Alt}{/Control}');
    expect(claimed).toEqual([]);
    expect(document.activeElement).toBe(textarea);

    await press('{Control>}{Meta>}{ArrowRight}{/Meta}{/Control}');
    expect(claimed).toEqual(['ctrl+meta+ArrowRight']);
    await vi.waitFor(() => expect(focusedRegion()).toBe(regionElement('center')));
    expect(textarea.value).toBe('two words');
  } finally {
    window.removeEventListener('keydown', observe);
  }
});
