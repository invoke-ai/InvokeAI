import { Box, chakra, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { useEntityAdapter } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import { debounce } from 'lodash-es';
import { memo, useEffect, useMemo, useRef } from 'react';
import { useSelector } from 'react-redux';

const ChakraCanvas = chakra.canvas;

const CONTAINER_WIDTH = 36; // this is size 12 in our theme - need it in px for the canvas
const CONTAINER_WIDTH_PX = `${CONTAINER_WIDTH}px`;

export const CanvasEntityPreviewImage = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapter(entityIdentifier);
  const selectMaskColor = useMemo(
    () =>
      createSelector(selectCanvasSlice, (state) => {
        const entity = selectEntity(state, entityIdentifier);
        if (!entity) {
          return null;
        }
        if (entity.type === 'inpaint_mask' || entity.type === 'regional_guidance') {
          return rgbColorToString(entity.fill.color);
        }
        return null;
      }),
    [entityIdentifier]
  );
  const maskColor = useSelector(selectMaskColor);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pixelRect = useStore(adapter.transformer.$pixelRect);
  const nodeRect = useStore(adapter.transformer.$nodeRect);
  const canvasCache = useStore(adapter.$canvasCache);

  const updatePreview = useMemo(
    () =>
      debounce(() => {
        if (!canvasRef.current) {
          return;
        }
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) {
          return;
        }

        const pixelRect = adapter.transformer.$pixelRect.get();
        const nodeRect = adapter.transformer.$nodeRect.get();
        const canvasCache = adapter.$canvasCache.get();

        if (!canvasCache || canvasCache.width === 0 || canvasCache.height === 0) {
          // Draw an empty canvas
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          return;
        }

        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        canvasRef.current.width = pixelRect.width;
        canvasRef.current.height = pixelRect.height;

        const sx = pixelRect.x - nodeRect.x;
        const sy = pixelRect.y - nodeRect.y;
        const sWidth = pixelRect.width;
        const sHeight = pixelRect.height;
        const dx = 0;
        const dy = 0;
        const dWidth = pixelRect.width;
        const dHeight = pixelRect.height;

        ctx.drawImage(canvasCache, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);

        if (maskColor) {
          ctx.fillStyle = maskColor;
          ctx.globalCompositeOperation = 'source-in';
          ctx.fillRect(0, 0, pixelRect.width, pixelRect.height);
        }
      }, 300),
    [adapter.$canvasCache, adapter.transformer.$nodeRect, adapter.transformer.$pixelRect, maskColor]
  );

  useEffect(updatePreview, [updatePreview, canvasCache, nodeRect, pixelRect]);

  return (
    <Flex
      position="relative"
      alignItems="center"
      justifyContent="center"
      w={CONTAINER_WIDTH_PX}
      h={CONTAINER_WIDTH_PX}
      borderRadius="sm"
      borderWidth={1}
      bg="base.900"
      flexShrink={0}
    >
      <Box
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        bgImage={TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL}
        bgSize="5px"
      />
      <ChakraCanvas position="relative" ref={canvasRef} objectFit="contain" maxW="full" maxH="full" />
    </Flex>
  );
});

CanvasEntityPreviewImage.displayName = 'CanvasEntityPreviewImage';
