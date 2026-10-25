import { openWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useOptionalWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { useCallback } from 'react';

import { requestLayerProperties } from './layerPropertiesRequestStore';

/**
 * Brings a layer's Properties into view: the Canvas in the center view, the Layers widget open, and the layer revealed
 * there once its row renders. Focus stays with the caller, so a toolbar or menu that asked keeps its place.
 */
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
