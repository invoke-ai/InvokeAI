import type {
  CanvasCoreStoreCapability,
  CanvasLayerContract,
  CanvasPreviewCapability,
  LayerThumbnailFallbackStage,
} from '@workbench/canvas-engine/api';
import type { CSSProperties } from 'react';

import { Box, Icon, IconButton } from '@chakra-ui/react';
import { galleryImageUrls } from '@features/gallery/utility';
import {
  getLayerThumbnailFallbackRenderState,
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
} from '@workbench/canvas-engine/api';
import { useLayerThumbnailStatus, useLayerThumbnailVersion } from '@workbench/widgets/canvas/engineStoreHooks';
import { ImageOffIcon, RefreshCwIcon } from 'lucide-react';
import { useCallback, useState } from 'react';

/** Backing-store cap for the thumbnail canvas (kept ≤128px per the engine contract). */
const THUMBNAIL_MAX_PX = 96;

const CANVAS_STYLE: CSSProperties = { height: '100%', objectFit: 'contain', width: '100%' };
const HIDDEN_STYLE: CSSProperties = { display: 'none' };
const IMG_STYLE: CSSProperties = { height: '100%', objectFit: 'cover', width: '100%' };

export type LayerThumbnailEngine = CanvasCoreStoreCapability & {
  readonly previews: CanvasPreviewCapability;
  readonly projectId: string;
};

/** Redraw live engine thumbnails by version; use persisted thumbnails or icons before cache availability. */
const LayerThumbnailContent = ({
  engine,
  layer,
}: {
  engine: LayerThumbnailEngine | null;
  layer: CanvasLayerContract;
}) => {
  // Each repaint keys a fresh canvas below, which binds and so re-blits the cache.
  const version = useLayerThumbnailVersion(engine, layer.id);
  const status = useLayerThumbnailStatus(engine, layer.id);
  const [drawn, setDrawn] = useState(false);
  const [fallbackStage, setFallbackStage] = useState<LayerThumbnailFallbackStage>('thumbnail');

  const bindCanvas = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas || !engine) {
        setDrawn(false);
        return;
      }
      if (status === 'idle') {
        void engine.previews.requestLayerThumbnail(layer.id);
      }
      const didDraw = engine.previews.drawLayerThumbnail(layer.id, canvas, THUMBNAIL_MAX_PX);
      setDrawn(didDraw);
      if (didDraw) {
        setFallbackStage('thumbnail');
      }
    },
    [engine, layer.id, status]
  );

  const retry = useCallback(() => {
    setFallbackStage('thumbnail');
    if (engine) {
      void engine.previews.requestLayerThumbnail(layer.id);
    }
  }, [engine, layer.id]);

  const fallbackImage = resolveLayerThumbnailImageRef(layer);
  const onFallbackError = useCallback(() => setFallbackStage(nextLayerThumbnailFallbackStage), []);
  const fallbackUrl =
    fallbackImage && fallbackStage !== 'failed'
      ? fallbackStage === 'thumbnail'
        ? galleryImageUrls.thumbnail(fallbackImage.imageName)
        : galleryImageUrls.full(fallbackImage.imageName)
      : null;
  const { showFallback, showRetry } = getLayerThumbnailFallbackRenderState(drawn, fallbackStage, status === 'error');

  return (
    <Box
      bg="bg.emphasized"
      borderColor="border.subtle"
      borderWidth="1px"
      flexShrink={0}
      h="full"
      overflow="hidden"
      position="relative"
      rounded="sm"
      w="full"
    >
      {/* A ref callback runs only when its element or identity changes, so the version keys the element. */}
      <canvas key={version ?? 'none'} ref={bindCanvas} style={drawn ? CANVAS_STYLE : HIDDEN_STYLE} />
      {showFallback &&
        (fallbackUrl ? (
          <img alt={layer.name} onError={onFallbackError} src={fallbackUrl} style={IMG_STYLE} />
        ) : (
          <Box alignItems="center" color="fg.subtle" display="flex" h="full" justifyContent="center" w="full">
            <Icon as={ImageOffIcon} boxSize="3.5" />
          </Box>
        ))}
      {showRetry ? (
        <IconButton
          aria-label={`Retry thumbnail for ${layer.name}`}
          inset="0"
          minH="0"
          minW="0"
          onClick={retry}
          onPointerDown={stopPropagation}
          position="absolute"
          variant="surface"
        >
          <RefreshCwIcon />
        </IconButton>
      ) : null}
    </Box>
  );
};

/** Key the boundary by project/layer/image to reset draw and fallback state on row reuse. */
export const LayerThumbnail = ({
  engine,
  layer,
}: {
  engine: LayerThumbnailEngine | null;
  layer: CanvasLayerContract;
}) => {
  const imageName = resolveLayerThumbnailImageRef(layer)?.imageName ?? 'none';
  return (
    <LayerThumbnailContent
      key={`${engine?.projectId ?? 'none'}:${layer.id}:${imageName}`}
      engine={engine}
      layer={layer}
    />
  );
};

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();
