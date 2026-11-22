import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { useCallback } from 'react';

import { requestLayerProperties } from './layerPropertiesRequestStore';

/** Brings a layer's Properties into view beside the Canvas; focus stays with the caller unless `focus` asks for it. */
export const useRevealLayerProperties = (): ((layerId: string, options?: { focus?: boolean }) => void) => {
  const open = useWorkspaceOpen();

  return useCallback(
    (layerId: string, { focus = false }: { focus?: boolean } = {}) => {
      open(
        [
          { surface: 'stage', typeId: 'canvas' },
          { surface: 'panel', typeId: 'layers' },
        ],
        { afterShown: () => requestLayerProperties(layerId), focus }
      );
    },
    [open]
  );
};
