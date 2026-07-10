/* oxlint-disable react-perf/jsx-no-new-function-as-prop -- the canvas ref callback is memoized on [engine, layerId, version]; it is intentionally re-created when the layer's thumbnail version bumps so the cache is re-blitted. */
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { LayerThumbnailFallbackStage } from '@workbench/canvas-engine/render/thumbnail';
import type { CanvasLayerContract } from '@workbench/types';
import type { CSSProperties } from 'react';

import { Box, Icon, IconButton } from '@chakra-ui/react';
import {
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
} from '@workbench/canvas-engine/render/thumbnail';
import { getImageFullUrl, getImageThumbnailUrl } from '@workbench/gallery/api';
import { useLayerThumbnailStatus, useLayerThumbnailVersion } from '@workbench/widgets/canvas/engineStoreHooks';
import { ImageOffIcon, RefreshCwIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

/** Backing-store cap for the thumbnail canvas (kept ≤128px per the engine contract). */
const THUMBNAIL_MAX_PX = 96;

const CANVAS_STYLE: CSSProperties = { height: '100%', objectFit: 'contain', width: '100%' };
const IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'cover', width: '100%' };

/**
 * A layer's thumbnail: the engine's live cache pixels drawn onto a `<canvas>`,
 * redrawn whenever the layer's `thumbnailVersion` bumps. Falls back to the
 * persisted image thumbnail (for image-source layers) or a placeholder icon
 * when there is no engine / no cache yet.
 */
const LayerThumbnailContent = ({ engine, layer }: { engine: CanvasEngine | null; layer: CanvasLayerContract }) => {
  // Re-renders (and thus re-runs the ref callback below) when the cache repaints.
  const version = useLayerThumbnailVersion(engine, layer.id);
  const status = useLayerThumbnailStatus(engine, layer.id);
  const [drawn, setDrawn] = useState(false);
  const [fallbackStage, setFallbackStage] = useState<LayerThumbnailFallbackStage>('thumbnail');

  const bindCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      // `version` is read purely so a repaint (a new version) re-creates this
      // callback, which React re-runs to re-blit the layer cache onto the canvas.
      void version;
      if (!canvas || !engine) {
        setDrawn(false);
        return;
      }
      if (status === 'idle') {
        void engine.requestLayerThumbnail(layer.id);
      }
      setDrawn(engine.drawLayerThumbnail(layer.id, canvas, THUMBNAIL_MAX_PX));
    },
    // `version` is a deliberate identity trigger: a repaint bumps it, giving the
    // callback a new identity so React re-runs it and re-blits the cache.
    // react-compiler can't model that implicit use, so it is suppressed here.
    // eslint-disable-next-line react/react-compiler
    [engine, layer.id, status, version]
  );

  const retry = useCallback(() => {
    setFallbackStage('thumbnail');
    if (engine) {
      void engine.requestLayerThumbnail(layer.id);
    }
  }, [engine, layer.id]);

  const fallbackImage = resolveLayerThumbnailImageRef(layer);
  const onFallbackError = useCallback(() => setFallbackStage(nextLayerThumbnailFallbackStage), []);
  const fallbackUrl =
    fallbackImage && fallbackStage !== 'failed'
      ? fallbackStage === 'thumbnail'
        ? getImageThumbnailUrl(fallbackImage.imageName)
        : getImageFullUrl(fallbackImage.imageName)
      : null;

  return (
    <Box
      bg="bg.emphasized"
      borderColor="border.subtle"
      borderWidth="1px"
      flexShrink={0}
      h="9"
      overflow="hidden"
      position="relative"
      rounded="sm"
      w="9"
    >
      <canvas ref={bindCanvas} style={drawn ? CANVAS_STYLE : { display: 'none' }} />
      {!drawn &&
        (fallbackUrl ? (
          <img alt={layer.name} onError={onFallbackError} src={fallbackUrl} style={IMG_STYLE} />
        ) : (
          <Box alignItems="center" color="fg.subtle" display="flex" h="full" justifyContent="center" w="full">
            <Icon as={ImageOffIcon} boxSize="3.5" />
          </Box>
        ))}
      {status === 'error' || fallbackStage === 'failed' ? (
        <IconButton
          aria-label={`Retry thumbnail for ${layer.name}`}
          inset="0"
          minH="0"
          minW="0"
          onClick={retry}
          position="absolute"
          size="xs"
          variant="surface"
        >
          <RefreshCwIcon />
        </IconButton>
      ) : null}
    </Box>
  );
};

/**
 * The keyed boundary resets local draw/fallback state when a row is reused for a
 * different project, layer, or persisted image. No effect-based reset is needed.
 */
export const LayerThumbnail = ({ engine, layer }: { engine: CanvasEngine | null; layer: CanvasLayerContract }) => {
  const imageName = resolveLayerThumbnailImageRef(layer)?.imageName ?? 'none';
  return (
    <LayerThumbnailContent
      key={`${engine?.projectId ?? 'none'}:${layer.id}:${imageName}`}
      engine={engine}
      layer={layer}
    />
  );
};
