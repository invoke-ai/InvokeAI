import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget } from '@workbench/widgetContracts';

import { canOpenWidget } from '@workbench/shell/widgetsListModel';

/** Widgets the palette offers, listed without loading any; the Workflow editor has its own command. */
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
