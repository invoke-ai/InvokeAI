import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { WidgetCommandApi, WidgetHotkeyApi } from '@workbench/widgetContracts';

import { useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

// The canvas history's undo/redo, reachable while a companion widget owns
// focus: selecting a layer or editing its properties must not strand mod+Z.
const HISTORY_HOTKEYS = [
  ['canvas.undo', 'widgets.canvas.commands.undo', ['mod+z']],
  ['canvas.redo', 'widgets.canvas.commands.redo', ['mod+shift+z', 'mod+y']],
] as const;

type HistoryCommandId = (typeof HISTORY_HOTKEYS)[number][0];

/** Registers canvas undo/redo on `runtime` for as long as the caller is mounted. */
export const useCanvasHistoryHotkeys = (
  runtime: { commands: Pick<WidgetCommandApi, 'register'>; hotkeys: WidgetHotkeyApi },
  engine: Pick<CanvasEngineHandle, 'history'> | null
): void => {
  const { t } = useTranslation();
  const run = useEffectEvent((commandId: HistoryCommandId) => {
    if (commandId === 'canvas.undo') {
      engine?.history.undo();
    } else {
      engine?.history.redo();
    }
  });
  useEffect(() => {
    const disposers = HISTORY_HOTKEYS.flatMap(([id, titleKey, defaultKeys]) => [
      runtime.commands.register({ handler: () => run(id), id, title: t(titleKey) }),
      runtime.hotkeys.register({
        allowInEditable: false,
        commandId: id,
        defaultKeys: [...defaultKeys],
        id,
        title: t(titleKey),
      }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);
};
