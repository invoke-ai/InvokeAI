/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type { WidgetContributionSource } from '@workbench/widgetContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider, createListCollection, Menu, Portal, Select } from '@chakra-ui/react';
import { PromptTextarea } from '@features/generation/components';
import { adjustFocusedPromptAttention } from '@features/generation/react';
import { MenuContent } from '@platform/ui/Menu';
import { AppToaster, createActionToast, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFloatingWindowFocus, useFocusRegionProps } from '@workbench/focusRegions';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useCallback, useState, type ChangeEvent } from 'react';
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

import { IS_MAC_OS } from './keys';
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
/** Mod+Comma, Settings: a global chord the popups leave alone (Zag menus swallow Enter whatever the modifiers). */
const pressOpenSettings = (target: EventTarget) =>
  act(async () => {
    target.dispatchEvent(
      new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        code: 'Comma',
        key: ',',
        ...(IS_MAC_OS ? { metaKey: true } : { ctrlKey: true }),
      })
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

describe('WorkbenchHotkeyRuntime and selected-image shortcuts', () => {
  /** A docked widget's DOM as the widget frame marks it for key targeting. */
  const mountWidget = (typeId: string, region: string) => {
    const element = document.createElement('div');
    element.dataset.hotkeyWidgetInstanceId = typeId;
    element.dataset.hotkeyWidgetRegion = region;
    element.dataset.hotkeyWidgetTypeId = typeId;
    host!.append(element);
    return element;
  };

  beforeEach(() => {
    // First-party commands are registered without a source, as `useRegisterFirstPartyCommands` does.
    for (const id of [
      'viewer.recallAll',
      'viewer.recallSeed',
      'viewer.recallPrompts',
      'viewer.useSize',
      'viewer.loadWorkflow',
      'viewer.remix',
      'gallery.remix',
    ]) {
      runtimeMocks.extensions.commands.register({ handler: () => ran.push(id), id, title: id });
    }
  });

  // Recall acts on the gallery selection, so its keys answer where the user selects it as well as in Preview.
  it('recalls from the gallery selection with focus in Gallery or Preview', async () => {
    const gallery = mountWidget('gallery', 'right');
    const preview = mountWidget('preview', 'center');

    await pressKey('a', gallery);
    await pressKey('s', gallery);
    await pressKey('p', gallery);
    await pressKey('d', gallery);
    await pressKey('w', gallery);
    await pressKey('s', preview);

    expect(ran).toEqual([
      'viewer.recallAll',
      'viewer.recallSeed',
      'viewer.recallPrompts',
      'viewer.useSize',
      'viewer.loadWorkflow',
      'viewer.recallSeed',
    ]);
  });

  it('keeps remix on each widget’s own binding and recall out of other widgets', async () => {
    const gallery = mountWidget('gallery', 'right');
    const preview = mountWidget('preview', 'center');

    await pressKey('r', gallery);
    await pressKey('r', preview);
    await pressKey('s', host!.querySelector('[data-testid="window"]')!);

    expect(ran).toEqual(['gallery.remix', 'viewer.remix']);
  });
});

describe('WorkbenchHotkeyRuntime and open menus', () => {
  let menuRoot: Root | null = null;

  beforeEach(() => {
    // First-party commands are registered without a source, as `useRegisterFirstPartyCommands` does.
    for (const id of ['viewer.recallSeed', 'app.openSettings']) {
      runtimeMocks.extensions.commands.register({ handler: () => ran.push(id), id, title: id });
    }
  });

  afterEach(async () => {
    await act(() => menuRoot?.unmount());
    menuRoot = null;
  });

  /** A gallery item's context menu: portaled content, so its DOM lies outside every widget. */
  const ImageMenu = () => (
    <Menu.Root>
      <Menu.Trigger>Image actions</Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent>
            <Menu.Item value="select">Select</Menu.Item>
            <Menu.Item value="star">Star</Menu.Item>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );

  const sizes = createListCollection({ items: ['Small', 'Seed', 'Square'] });
  /** A settings Select: portaled content that takes focus as a `listbox` and runs its own typeahead. */
  const SizeSelect = () => (
    <Select.Root collection={sizes}>
      <Select.Control>
        <Select.Trigger>
          <Select.ValueText placeholder="Size" />
        </Select.Trigger>
      </Select.Control>
      <Portal>
        <Select.Positioner>
          <Select.Content>
            {sizes.items.map((item) => (
              <Select.Item key={item} item={item}>
                {item}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Positioner>
      </Portal>
    </Select.Root>
  );

  /** Gallery holds workbench focus, then the popup opens from its trigger and takes focus. */
  const openPopup = async (popup: React.ReactNode, role: 'listbox' | 'menu') => {
    await pressOn('right');
    const menuHost = document.createElement('div');
    host!.append(menuHost);
    menuRoot = createRoot(menuHost);
    await act(() => {
      menuRoot?.render(<ChakraProvider value={system}>{popup}</ChakraProvider>);
    });
    await act(() => userEvent.click(menuHost.querySelector('button')!));
    const content = await vi.waitFor(() => {
      const element = document.querySelector<HTMLElement>(`[role="${role}"][data-state="open"]`);
      expect(element).not.toBeNull();
      return element!;
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(content));
    return content;
  };

  it('still runs a global chord pressed inside a portaled menu', async () => {
    const menu = await openPopup(<ImageMenu />, 'menu');

    await pressKey('s', menu);
    await pressOpenSettings(menu);

    expect(ran).toEqual(['app.openSettings']);
  });

  it('leaves a letter typed into a portaled Select listbox to the Select while Gallery holds focus', async () => {
    const listbox = await openPopup(<SizeSelect />, 'listbox');

    await pressKey('s', listbox);
    expect(ran).toEqual([]);
    await pressOpenSettings(listbox);
    expect(ran).toEqual(['app.openSettings']);
  });

  it('leaves a letter typed into a portaled menu to the menu while Gallery holds focus', async () => {
    // Gallery is the right region's active widget: a press outside any widget's DOM resolves against it.
    await pressOn('right');
    await pressKey('s');
    expect(ran).toEqual(['viewer.recallSeed']);

    const menuHost = document.createElement('div');
    host!.append(menuHost);
    menuRoot = createRoot(menuHost);
    await act(() => {
      menuRoot?.render(
        <ChakraProvider value={system}>
          <ImageMenu />
        </ChakraProvider>
      );
    });
    await act(() => userEvent.click(menuHost.querySelector('button')!));
    const menu = await vi.waitFor(() => {
      const content = document.querySelector<HTMLElement>('[role="menu"][data-state="open"]');
      expect(content).not.toBeNull();
      return content!;
    });
    // Zag focuses the content on open; typeahead keydowns target it, prevent their default, and still bubble.
    await vi.waitFor(() => expect(document.activeElement).toBe(menu));

    await pressKey('s', menu);
    await pressKey('s', menu.querySelector('[role="menuitem"]')!);

    expect(ran).toEqual(['viewer.recallSeed']);
    // Gallery still holds workbench focus: the menu, not lost focus, kept the key.
    await pressKey('s');
    expect(ran).toEqual(['viewer.recallSeed', 'viewer.recallSeed']);
  });
});

describe('WorkbenchHotkeyRuntime and prompt weighting', () => {
  let promptRoot: Root | null = null;

  beforeEach(() => {
    // Registered without a source, as `useRegisterFirstPartyCommands` does.
    runtimeMocks.extensions.commands.register({
      handler: () => adjustFocusedPromptAttention('increment', false),
      id: 'app.promptWeightUp',
      title: 'app.promptWeightUp',
    });
  });

  afterEach(async () => {
    await act(() => promptRoot?.unmount());
    promptRoot = null;
  });

  const PromptHarness = ({ attentionSyntax }: { attentionSyntax: 'edit' | 'none' }) => {
    const [value, setValue] = useState('a cat');
    const handleChange = useCallback(
      (event: ChangeEvent<HTMLTextAreaElement>) => setValue(event.currentTarget.value),
      []
    );

    return (
      <PromptTextarea
        aria-label="Prompt"
        attentionSyntax={attentionSyntax}
        defaultHeightPx={96}
        minHeightPx={56}
        resizeHandleAriaLabel="Resize prompt"
        showSyntaxHighlighting
        value={value}
        onChange={handleChange}
      />
    );
  };

  /** Select "cat" and press Ctrl+Up; report whether the runtime took the key from the text field. */
  const pressWeightUp = async (attentionSyntax: 'edit' | 'none') => {
    const promptHost = document.createElement('div');
    host!.append(promptHost);
    promptRoot = createRoot(promptHost);
    await act(() => {
      promptRoot?.render(
        <ChakraProvider value={system}>
          <PromptHarness attentionSyntax={attentionSyntax} />
        </ChakraProvider>
      );
    });
    const textarea = promptHost.querySelector('textarea')!;
    textarea.focus();
    textarea.setSelectionRange(2, 5);

    let claimed = false;
    await act(async () => {
      claimed = !textarea.dispatchEvent(
        new KeyboardEvent('keydown', {
          bubbles: true,
          cancelable: true,
          code: 'ArrowUp',
          ctrlKey: true,
          key: 'ArrowUp',
        })
      );
      await Promise.resolve();
    });

    return { claimed, textarea };
  };

  it('weights the selection in a prompt whose model reads attention syntax', async () => {
    const { claimed, textarea } = await pressWeightUp('edit');

    expect(claimed).toBe(true);
    await expect.poll(() => textarea.value).toBe('a cat+');
  });

  it('leaves Ctrl+Up to the text field when the model reads the prompt as plain text', async () => {
    const { claimed, textarea } = await pressWeightUp('none');

    expect(claimed).toBe(false);
    expect(textarea.value).toBe('a cat');
  });
});
