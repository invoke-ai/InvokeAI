import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { WidgetCommandApi, WidgetHotkeyApi } from '@workbench/widgetContracts';

import { useNotify } from '@workbench/useNotify';
import {
  CANVAS_COMPANION_HOTKEYS,
  CANVAS_HISTORY_HOTKEYS,
  CANVAS_MERGE_DOWN_HOTKEY,
  executeCanvasToolCommand,
  mergeSelectedLayerDown,
  reportMergeDownRefusal,
  reportTransformRefusal,
  showCanvasRefusalNotice,
  type CanvasToolCommandId,
} from '@workbench/widgets/canvas/canvasToolCommands';
import { useActiveColorCommands } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { reportLayerOperation } from '@workbench/widgets/canvas/useStructuralCommit';
import { useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

type CompanionCommandId = (typeof CANVAS_HISTORY_HOTKEYS)[number][0] | CanvasToolCommandId | 'canvas.mergeDown';

export type CanvasCompanionEngine = Pick<
  CanvasEngineHandle,
  'document' | 'history' | 'interaction' | 'layers' | 'tools'
>;

/**
 * Registers the canvas keys that stay live while a companion widget (Layers and its Color pane) holds focus:
 * undo/redo, merge down, and the shared tool, brush-size and color table. Rows keep arrows, Delete and selection.
 */
export const useCanvasCompanionHotkeys = (
  runtime: { commands: Pick<WidgetCommandApi, 'register'>; hotkeys: WidgetHotkeyApi },
  engine: CanvasCompanionEngine | null
): void => {
  const { t } = useTranslation();
  const notify = useNotify();
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
    const model = engine?.document.model() ?? null;
    if (commandId === 'canvas.mergeDown') {
      if (model) {
        mergeSelectedLayerDown(
          engine,
          model.document,
          (refusal) => reportLayerOperation(refusal, notify.error, t),
          (refusal) => reportMergeDownRefusal(model.document, refusal, showCanvasRefusalNotice, t)
        );
      }
      return;
    }
    const selectedId = model?.document.selectedLayerId ?? null;
    executeCanvasToolCommand(commandId, {
      engine,
      reportTransformRefusal: (refusal, layerId) =>
        reportTransformRefusal(
          refusal,
          { id: layerId, name: model?.getLayer(layerId)?.name ?? '' },
          { error: notify.error, notice: showCanvasRefusalNotice },
          t
        ),
      resetActiveColors: colorCommands.resetPair,
      selectedLayerId: selectedId === null ? null : (model?.getLayer(selectedId)?.id ?? null),
      swapActiveColors: colorCommands.swapPair,
    });
  });
  useEffect(() => {
    // Undo/redo stay reachable while a companion owns focus: selecting a layer or editing it must not strand mod+Z.
    const disposers = [...CANVAS_HISTORY_HOTKEYS, ...CANVAS_COMPANION_HOTKEYS, CANVAS_MERGE_DOWN_HOTKEY].flatMap(
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
