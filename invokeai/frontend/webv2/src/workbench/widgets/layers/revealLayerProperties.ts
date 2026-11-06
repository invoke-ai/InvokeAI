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

/** How long a reveal waits for the Properties pane, just opened, to render the Layer section. */
const REVEAL_WAIT_MS = 1500;

/**
 * Scrolls the Properties pane to its Layer section, which sits below the active tool's settings and often starts out of
 * view. The caller opens the pane first; a reveal whose pane never renders gives up quietly.
 */
export const revealLayerSection = (): void => {
  const deadline = performance.now() + REVEAL_WAIT_MS;

  const attempt = () => {
    const section = [...document.querySelectorAll<HTMLElement>('[data-properties-section="layer"]')].find(
      (element) => element.getClientRects().length > 0
    );
    if (section) {
      section.scrollIntoView({ block: 'nearest' });
    } else if (performance.now() < deadline) {
      requestAnimationFrame(attempt);
    }
  };

  requestAnimationFrame(attempt);
};
