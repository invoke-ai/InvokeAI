import { openWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useOptionalWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback } from 'react';

import { requestLayerProperties } from './layerPropertiesRequestStore';

/** Brings a layer's Properties into view; focus stays with the caller. */
export const useRevealLayerProperties = (): ((layerId: string) => void) => {
  const { widgets } = useWorkbenchCommands();
  const registry = useOptionalWorkbenchWidgetRegistry();

  return useCallback(
    (layerId: string) => {
      if (registry) {
        openWorkbenchWidget(widgets, registry.getWidgetsForRegion, 'canvas', {
          preferredRegions: ['center'],
          requireCenterView: true,
        });
      }
      widgets.open({ region: 'right', widgetId: 'layers' });
      requestLayerProperties(layerId);
    },
    [registry, widgets]
  );
};
