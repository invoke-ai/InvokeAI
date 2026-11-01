import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget } from '@workbench/widgetContracts';

import { canOpenWidget } from '@workbench/shell/widgetsListModel';

/**
 * The widgets the palette offers to show: each enabled widget a placement open can show, once, without loading any of
 * them (the palette lists them on every open; only the entry chosen loads its widget). The Workflow editor has its own
 * command, which opens it in the center, so it is offered once, there.
 */
export const listPaletteWidgets = (
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[]
): RegisteredWidget[] => {
  const seen = new Set<string>();

  return (['center', 'left', 'right', 'bottom'] as const)
    .flatMap((region) => getWidgetsForRegion(region))
    .filter((widget) => {
      const typeId = widget.manifest.id;

      if (seen.has(typeId)) {
        return false;
      }

      seen.add(typeId);
      return typeId !== 'workflow' && canOpenWidget(widget, getWidgetsForRegion);
    });
};
