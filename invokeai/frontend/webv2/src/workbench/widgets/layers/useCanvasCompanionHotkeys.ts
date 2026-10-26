import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { WidgetCommandApi, WidgetHotkeyApi } from '@workbench/widgetContracts';

import { useNotify } from '@workbench/useNotify';
import { isCanvasInteractionLocked } from '@workbench/widgets/canvas/canvasInteractionLock';
import {
  CANVAS_COMPANION_HOTKEYS,
  CANVAS_HISTORY_HOTKEYS,
  executeCanvasToolCommand,
  reportTransformRefusal,
  type CanvasToolCommandId,
} from '@workbench/widgets/canvas/canvasToolCommands';
import { useActiveColorCommands } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

type CompanionCommandId = (typeof CANVAS_HISTORY_HOTKEYS)[number][0] | CanvasToolCommandId;

export type CanvasCompanionEngine = Pick<
  CanvasEngineHandle,
  'document' | 'history' | 'interaction' | 'layers' | 'tools'
>;

/**
 * Registers the canvas keys that stay live while a companion widget (Layers and its Color pane) holds focus:
 * undo/redo and the shared tool, brush-size and color table. Rows keep arrows, Delete and selection.
 */
export const useCanvasCompanionHotkeys = (
  runtime: { commands: Pick<WidgetCommandApi, 'register'>; hotkeys: WidgetHotkeyApi },
  engine: CanvasCompanionEngine | null
): void => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queries = useWorkbenchQueries();
  const colorCommands = useActiveColorCommands();
  const run = useEffectEvent((commandId: CompanionCommandId) => {
    if (commandId === 'canvas.undo') {
      engine?.history.undo();
      return;
    }
    if (commandId === 'canvas.redo') {
      engine?.history.redo();
      return;
    }
    const { canvas, queue } = queries.getSnapshot().activeProject;
    const model = engine?.document.model() ?? null;
    const selectedId = model?.document.selectedLayerId ?? null;
    executeCanvasToolCommand(commandId, {
      engine,
      isInteractionLocked: isCanvasInteractionLocked(canvas, queue.items),
      reportTransformRefusal: (refusal) => reportTransformRefusal(refusal, notify.error, t),
      resetActiveColors: colorCommands.resetPair,
      selectedLayerId: selectedId === null ? null : (model?.getLayer(selectedId)?.id ?? null),
      swapActiveColors: colorCommands.swapPair,
    });
  });
  useEffect(() => {
    // Undo/redo stay reachable while a companion owns focus: selecting a layer or editing it must not strand mod+Z.
    const disposers = [...CANVAS_HISTORY_HOTKEYS, ...CANVAS_COMPANION_HOTKEYS].flatMap(
      ([id, titleKey, defaultKeys]) => {
        const title = t(titleKey);
        return [
          runtime.commands.register({ handler: () => run(id), id, title }),
          runtime.hotkeys.register({ allowInEditable: false, commandId: id, defaultKeys: [...defaultKeys], id, title }),
        ];
      }
    );

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);
};
