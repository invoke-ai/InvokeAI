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

/** How long a reveal waits for the Properties pane, just opened, to render the Layer section. */
const REVEAL_WAIT_MS = 1500;

/**
 * Scrolls the Properties pane to the start of its Layer section, which sits below the active tool's settings and often
 * starts out of view. The caller opens the pane first; a reveal whose pane never renders gives up quietly.
 */
export const revealLayerSection = (): void => {
  const deadline = performance.now() + REVEAL_WAIT_MS;

  const attempt = () => {
    const section = [...document.querySelectorAll<HTMLElement>('[data-properties-section="layer"]')].find(
      (element) => element.getClientRects().length > 0
    );
    if (section) {
      // `start`, not `nearest`: a section taller than the pane, already scrolled into, would otherwise not move.
      section.scrollIntoView({ block: 'start' });
    } else if (performance.now() < deadline) {
      requestAnimationFrame(attempt);
    }
  };

  requestAnimationFrame(attempt);
};
