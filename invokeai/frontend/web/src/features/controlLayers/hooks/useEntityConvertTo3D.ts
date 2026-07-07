import { logger } from 'app/logging/logger';
import { withResultAsync } from 'common/util/result';
import {
  $splatOverlay,
  clearSplatGenerationAbort,
  clearSplatOverlay,
  setSplatGenerationAbort,
} from 'features/controlLayers/components/SplatOverlay/state';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsEmpty } from 'features/controlLayers/hooks/useEntityIsEmpty';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { useCallback, useMemo } from 'react';
import { buildV1Url } from 'services/api';

const log = logger('canvas');

/**
 * Trigger for the "Convert to 3D" raster-layer action: rasterizes the layer, runs the `image_to_3d`
 * (TripoSplat) node, and opens the splat overlay with the resulting .ply. Mirrors useEntitySegmentAnything.
 */
export const useEntityConvertTo3D = (entityIdentifier: CanvasEntityIdentifier | null) => {
  const canvasManager = useCanvasManager();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isLocked = useEntityIsLocked(entityIdentifier);
  const isEmpty = useEntityIsEmpty(entityIdentifier);

  const isDisabled = useMemo(() => {
    if (!entityIdentifier || entityIdentifier.type !== 'raster_layer') {
      return true;
    }
    if (!adapter || isBusy || isLocked || isEmpty) {
      return true;
    }
    return false;
  }, [entityIdentifier, adapter, isBusy, isLocked, isEmpty]);

  const start = useCallback(
    (removeBackground: boolean) => {
      if (isDisabled || !entityIdentifier || entityIdentifier.type !== 'raster_layer') {
        return;
      }
      const adapter = canvasManager.getAdapter(entityIdentifier);
      if (!adapter) {
        return;
      }
      const rect = adapter.transformer.getRelativeRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }

      // Only one overlay session can exist; replacing a session must abort its backend run first, or the
      // old run becomes uncancelable and its late completion would write into the new session's state.
      clearSplatOverlay();
      const sessionId = getPrefixedId('splat_session');
      const controller = new AbortController();
      setSplatGenerationAbort(controller);
      $splatOverlay.set({ status: 'loading', sessionId, rect });

      void (async () => {
        const result = await withResultAsync(async () => {
          const imageDTO = await adapter.renderer.rasterize({ rect, attrs: { filters: [], opacity: 1 } });
          const graph = new Graph(getPrefixedId('canvas_image_to_3d'));
          const node = graph.addNode({
            id: getPrefixedId('image_to_3d'),
            type: 'image_to_3d',
            image: { image_name: imageDTO.image_name },
            remove_background: removeBackground,
          });
          const output = await canvasManager.stateApi.runGraphAndReturnOutput({
            graph,
            outputNodeId: node.id,
            options: { prepend: true, signal: controller.signal },
          });
          if (output.type !== 'asset_3d_output') {
            throw new Error(`Unexpected output type: ${output.type}`);
          }
          return buildV1Url(`assets/i/${output.asset.asset_name}`);
        });

        clearSplatGenerationAbort(controller);

        // Every write below is gated on the overlay still showing *this* session — the user may have
        // cancelled (state null) or started another conversion (different sessionId) while we generated.
        const current = $splatOverlay.get();
        const isCurrentSession = current?.status === 'loading' && current.sessionId === sessionId;

        if (result.isErr()) {
          log.error({ error: String(result.error) }, 'Failed to convert image to 3D');
          if (isCurrentSession) {
            clearSplatOverlay();
          }
          return;
        }

        if (isCurrentSession) {
          // Use the overlay's *current* rect, not the one captured at start — the frame is movable while loading.
          $splatOverlay.set({ status: 'ready', sessionId, assetUrl: result.value, rect: current.rect });
        }
      })();
    },
    [isDisabled, entityIdentifier, canvasManager]
  );

  return { isDisabled, start } as const;
};
