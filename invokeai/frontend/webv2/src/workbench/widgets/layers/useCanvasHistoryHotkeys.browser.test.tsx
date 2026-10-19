import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { WidgetCommandContribution, WidgetHotkeyContribution } from '@workbench/widgetContracts';

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useCanvasHistoryHotkeys } from './useCanvasHistoryHotkeys';

const translation = { t: (key: string) => key };
vi.mock('react-i18next', () => ({ useTranslation: () => translation }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createRuntime = () => {
  const commands = new Map<string, WidgetCommandContribution>();
  const hotkeys = new Map<string, WidgetHotkeyContribution>();
  return {
    commands: {
      register: vi.fn((command: WidgetCommandContribution) => {
        commands.set(command.id, command);
        return () => commands.delete(command.id);
      }),
    },
    hotkeys: {
      register: vi.fn((hotkey: WidgetHotkeyContribution) => {
        hotkeys.set(hotkey.id, hotkey);
        return () => hotkeys.delete(hotkey.id);
      }),
    },
    registered: { commands, hotkeys },
  };
};

const Host = ({
  engine,
  runtime,
}: {
  engine: Pick<CanvasEngineHandle, 'history'> | null;
  runtime: ReturnType<typeof createRuntime>;
}) => {
  useCanvasHistoryHotkeys(runtime, engine);
  return null;
};

let root: Root | null = null;
let host: HTMLDivElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const mount = (element: ReactElement): void => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  act(() => root!.render(element));
};

describe('useCanvasHistoryHotkeys', () => {
  it('registers undo/redo outside editable fields and drives the current engine', () => {
    const runtime = createRuntime();
    const first = { history: { redo: vi.fn(), undo: vi.fn() } } as unknown as CanvasEngineHandle;
    const second = { history: { redo: vi.fn(), undo: vi.fn() } } as unknown as CanvasEngineHandle;
    mount(createElement(Host, { engine: first, runtime }));

    expect([...runtime.registered.hotkeys.values()]).toEqual([
      expect.objectContaining({ allowInEditable: false, commandId: 'canvas.undo', defaultKeys: ['mod+z'] }),
      expect.objectContaining({
        allowInEditable: false,
        commandId: 'canvas.redo',
        defaultKeys: ['mod+shift+z', 'mod+y'],
      }),
    ]);
    void runtime.registered.commands.get('canvas.undo')!.handler();
    expect(first.history.undo).toHaveBeenCalledOnce();

    // A new engine identity re-renders the host without re-registering.
    act(() => root!.render(createElement(Host, { engine: second, runtime })));
    void runtime.registered.commands.get('canvas.redo')!.handler();
    expect(second.history.redo).toHaveBeenCalledOnce();
    expect(first.history.redo).not.toHaveBeenCalled();
    expect(runtime.commands.register).toHaveBeenCalledTimes(2);
  });

  it('withdraws both registrations on unmount and tolerates a missing engine', () => {
    const runtime = createRuntime();
    mount(createElement(Host, { engine: null, runtime }));
    expect(() => runtime.registered.commands.get('canvas.undo')!.handler()).not.toThrow();

    act(() => root!.unmount());
    expect(runtime.registered.commands.size).toBe(0);
    expect(runtime.registered.hotkeys.size).toBe(0);
  });
});
