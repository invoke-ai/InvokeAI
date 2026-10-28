import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type { WidgetContributionSource } from '@workbench/widgetContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { useFocusRegionProps } from '@workbench/focusRegions';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { readActiveColorPair } from '@workbench/widgets/canvas/color-system/colorPair';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { useCanvasCompanionHotkeys, type CanvasCompanionEngine } from './useCanvasCompanionHotkeys';

/**
 * The companion keys under the real hotkey runtime, focus provider, store and extension registry: a key pressed
 * with focus in the Layers widget reaches the canvas engine. Only context plumbing and first-party commands, which
 * need the whole application, are replaced.
 */
const mocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) =>
    selector(mocks.store.getSnapshot().activeProject),
  useOptionalWorkbenchCommands: () => mocks.store.commands,
  useOptionalWorkbenchExtensions: () => mocks.extensions,
  useWorkbenchCommands: () => mocks.store.commands,
  useWorkbenchExtensions: () => mocks.extensions,
  useWorkbenchInternalStore: () => mocks.store,
  useWorkbenchQueries: () => mocks.store.queries,
  useWorkbenchSubscription: () => mocks.store.subscribe,
}));
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createEngine = () => ({
  document: { model: () => null },
  history: { redo: vi.fn(), undo: vi.fn() },
  interaction: { get: vi.fn(() => 'brush'), set: vi.fn() },
  layers: { beginTransform: vi.fn(() => 'started') },
  tools: { setTool: vi.fn(), stepBrushSize: vi.fn() },
});

/** The Layers widget's frame as the shell renders it, registering through a widget runtime bound to its source. */
const LayersFrame = ({ engine }: { engine: ReturnType<typeof createEngine> }) => {
  const [runtime] = useState(() => {
    const source: WidgetContributionSource = {
      instanceId: 'layers',
      projectId: mocks.store.getSnapshot().activeProject.id,
      region: 'right',
      typeId: 'layers',
    };
    return {
      commands: { register: (command) => mocks.extensions.commands.register({ ...command, source }) },
      hotkeys: { register: (hotkey) => mocks.extensions.hotkeys.register({ ...hotkey, scope: 'widget', source }) },
    } satisfies Parameters<typeof useCanvasCompanionHotkeys>[0];
  });
  useCanvasCompanionHotkeys(runtime, engine as unknown as CanvasCompanionEngine);

  return (
    <div
      {...useFocusRegionProps('right')}
      data-hotkey-widget-instance-id="layers"
      data-hotkey-widget-region="right"
      data-hotkey-widget-type-id="layers"
    >
      <div role="tree">
        <div aria-selected="true" role="treeitem" tabIndex={0}>
          Layer 1
        </div>
      </div>
      <button type="button">Swatches</button>
      <input aria-label="Hex" />
    </div>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let engine = createEngine();

const colorPair = () => readActiveColorPair(getProjectWidgetValues(mocks.store.getSnapshot().activeProject, 'canvas'));

beforeEach(async () => {
  mocks.store = createWorkbenchStore();
  mocks.extensions = createExtensionRegistry();
  engine = createEngine();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <WorkbenchFocusProvider>
        <WorkbenchHotkeyRuntime />
        <LayersFrame engine={engine} />
      </WorkbenchFocusProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('useCanvasCompanionHotkeys', () => {
  it('selects canvas tools from a clicked layer row without taking focus from it', async () => {
    const row = host!.querySelector<HTMLElement>('[role="treeitem"]')!;
    await userEvent.click(row);

    await userEvent.keyboard('b');

    expect(engine.tools.setTool).toHaveBeenCalledExactlyOnceWith('brush');
    expect(document.activeElement).toBe(row);
  });

  it('swaps the color pair and steps the brush from a clicked control, and switches tools from it', async () => {
    const before = colorPair();
    await userEvent.click(host!.querySelector('button')!);

    await userEvent.keyboard('x');
    await userEvent.keyboard(']');
    await userEvent.keyboard('e');

    expect(colorPair()).toEqual({ background: before.foreground, foreground: before.background });
    expect(engine.tools.stepBrushSize).toHaveBeenCalledExactlyOnceWith(1);
    expect(engine.tools.setTool).toHaveBeenCalledExactlyOnceWith('eraser');
  });

  it('keeps undo and redo on the canvas history', async () => {
    await userEvent.click(host!.querySelector('[role="treeitem"]')!);

    await userEvent.keyboard(IS_MAC_OS ? '{Meta>}z{/Meta}' : '{Control>}z{/Control}');

    expect(engine.history.undo).toHaveBeenCalledOnce();
  });

  it('leaves typing in a field alone', async () => {
    const field = host!.querySelector('input')!;
    await userEvent.click(field);

    await userEvent.keyboard('b');

    expect(field.value).toBe('b');
    expect(engine.tools.setTool).not.toHaveBeenCalled();
  });

  it('withdraws every registration on unmount', async () => {
    await act(() => root!.unmount());
    root = null;

    expect(mocks.extensions.stores.hotkeys.list()).toEqual([]);
    expect(mocks.extensions.stores.commands.list()).toEqual([]);
  });
});
