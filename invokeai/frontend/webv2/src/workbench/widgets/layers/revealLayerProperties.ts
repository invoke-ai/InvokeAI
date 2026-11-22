import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { useCallback } from 'react';

import { requestLayerProperties } from './layerPropertiesRequestStore';

/**
 * Brings a layer's Properties into view beside the Canvas. Focus stays with the caller unless `focus` names the widget
 * to take it, wherever that shows: a rail, a window or a guest.
 */
export const useRevealLayerProperties = (): ((layerId: string, options?: { focus?: 'canvas' | 'layers' }) => void) => {
  const open = useWorkspaceOpen();

  return useCallback(
    (layerId: string, { focus }: { focus?: 'canvas' | 'layers' } = {}) => {
      open(
        [
          { surface: 'stage', typeId: 'canvas' },
          { surface: 'panel', typeId: 'layers' },
        ],
        { afterShown: () => requestLayerProperties(layerId), focus: focus ?? false }
      );
    },
    [open]
  );
};
