/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type { WidgetContributionSource } from '@workbench/widgetContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, createActionToast, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFloatingWindowFocus, useFocusRegionProps } from '@workbench/focusRegions';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

/**
 * The hotkey runtime under the real focus provider, a real store, and the real extension registry: which command
 * a key press runs depends on the floating window that holds focus. Only the context plumbing and the first-party
 * command registrations (which need the whole application) are replaced.
 */

const runtimeMocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) =>
    selector(runtimeMocks.store.getSnapshot().activeProject),
  useOptionalWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchExtensions: () => runtimeMocks.extensions,
  useWorkbenchInternalStore: () => runtimeMocks.store,
  useWorkbenchQueries: () => runtimeMocks.store.queries,
  useWorkbenchSubscription: () => runtimeMocks.store.subscribe,
}));
vi.mock('./firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

import { WorkbenchHotkeyRuntime } from './WorkbenchHotkeyRuntime';

const Harness = () => {
  const { activate } = useFloatingWindowFocus('image-map', runtimeMocks.store.getSnapshot().activeProject.id);
  const handlePointerDown = useCallback(() => activate({ byPointer: true }), [activate]);

  return (
    <>
      <div data-testid="right" {...useFocusRegionProps('right')}>
        Right panel
      </div>
      <div
        data-floating-window="image-map"
        data-hotkey-widget-instance-id="image-map"
        data-hotkey-widget-region="floating"
        data-hotkey-widget-type-id="image-map"
        data-testid="window"
        onPointerDownCapture={handlePointerDown}
      >
        <button type="button">Window control</button>
      </div>
    </>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let ran: string[] = [];

const projectId = () => runtimeMocks.store.getSnapshot().activeProject.id;
/** Register a command and a shortcut for it, the way a widget does through its runtime. */
const contribute = (
  id: string,
  key: string,
  scope: 'focused-region' | 'instance' | 'widget',
  source: WidgetContributionSource
) => {
  runtimeMocks.extensions.commands.register({ handler: () => ran.push(id), id, source, title: id });
  runtimeMocks.extensions.hotkeys.register({ commandId: id, defaultKeys: [key], id, scope, source, title: id });
};
const pressKey = (key: string, target: EventTarget = document.body) =>
  act(async () => {
    target.dispatchEvent(
      new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code: `Key${key.toUpperCase()}`, key })
    );
    await Promise.resolve();
  });
const pressOn = (testId: string) =>
  act(() =>
    host!.querySelector(`[data-testid="${testId}"]`)!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
  );

beforeEach(async () => {
  ran = [];
  runtimeMocks.store = createWorkbenchStore();
  runtimeMocks.store.commands.widgets.float('image-map');
  runtimeMocks.extensions = createExtensionRegistry();

  const floating: WidgetContributionSource = {
    instanceId: 'image-map',
    projectId: projectId(),
    region: 'floating',
    typeId: 'image-map',
  };
  const docked: WidgetContributionSource = {
    instanceId: 'gallery',
    projectId: projectId(),
    region: 'right',
    typeId: 'gallery',
  };

  // Keys the first-party catalog does not use.
  contribute('map.instance', 'j', 'instance', floating);
  contribute('map.widget', 'k', 'widget', floating);
  contribute('map.window', 'y', 'focused-region', floating);
  contribute('gallery.rightRegion', 'y', 'focused-region', docked);

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WorkbenchFocusProvider>
          <WorkbenchHotkeyRuntime />
          <Harness />
          <AppToaster />
        </WorkbenchFocusProvider>
      </ChakraProvider>
    );
  });
});

afterEach(async () => {
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchHotkeyRuntime under a floating window', () => {
  it('runs nothing widget-scoped while nothing holds focus', async () => {
    await pressKey('j');
    await pressKey('k');
    await pressKey('y');

    expect(ran).toEqual([]);
  });

  it('aims key presses outside any widget at the active floating window', async () => {
    await pressOn('window');

    await pressKey('j');
    await pressKey('k');
    await pressKey('y');

    // Instance, widget, and the window's own focused-region shortcut all run; the shortcut tied to the right
    // region shares the key and stays out of it.
    expect(ran).toEqual(['map.instance', 'map.widget', 'map.window']);
  });

  it('aims key presses inside the window at it, whatever focus last recorded', async () => {
    await pressOn('right');
    const control = host!.querySelector('button')!;

    await pressKey('j', control);
    await pressKey('y', control);

    expect(ran).toEqual(['map.instance', 'map.window']);
  });

  it('gives the shared key back to the docked region once that region holds focus', async () => {
    await pressOn('window');
    await pressOn('right');

    await pressKey('y');
    await pressKey('j');

    expect(ran).toEqual(['gallery.rightRegion']);
  });

  it('stops aiming at the window once it is closed', async () => {
    await pressOn('window');
    await act(() => runtimeMocks.store.commands.widgets.closeFloating('image-map'));

    await pressKey('j');
    await pressKey('y');

    expect(ran).toEqual([]);
  });
});

describe('WorkbenchHotkeyRuntime and toast buttons', () => {
  it('leaves Enter and Space on a focused toast button to the button, while a widget binds Enter', async () => {
    const floating: WidgetContributionSource = {
      instanceId: 'image-map',
      projectId: projectId(),
      region: 'floating',
      typeId: 'image-map',
    };
    // What an active canvas session's apply command looks like to the runtime.
    contribute('map.apply', 'enter', 'widget', floating);
    const retried = vi.fn();
    await pressOn('window');

    await act(() => {
      createActionToast({ actions: [{ label: 'Retry', onClick: retried }], title: 'Saved', type: 'warning' });
    });
    await act(() => vi.waitFor(() => expect(document.querySelector('[data-scope="toast"] button')).not.toBeNull()));
    const retry = [...document.querySelectorAll<HTMLButtonElement>('[data-scope="toast"] button')].find(
      (button) => button.textContent === 'Retry'
    )!;

    retry.focus();
    await act(() => userEvent.keyboard('{Enter}'));

    expect(retried).toHaveBeenCalledOnce();
    expect(ran).toEqual([]);

    // Outside the toast the binding still runs.
    await pressKey('Enter', host!.querySelector('[data-testid="window"]')!);
    expect(ran).toEqual(['map.apply']);
  });
});
