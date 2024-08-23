import { Box, chakra, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { rgbColorToString } from 'common/util/colorCodeTransformers';
import { useEntityAdapter } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { selectCanvasV2Slice, selectEntity } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useEffect, useMemo, useRef } from 'react';
import { useSelector } from 'react-redux';

const ChakraCanvas = chakra.canvas;

const PADDING = 4;

export const CanvasEntityPreviewImage = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapter();
  const selectMaskColor = useMemo(
    () =>
      createSelector(selectCanvasV2Slice, (state) => {
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
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cache = useStore(adapter.renderer.$canvasCache);
  useEffect(() => {
    if (!cache || !canvasRef.current || !containerRef.current) {
      return;
    }
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) {
      return;
    }
    const { rect, canvas } = cache;

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    canvasRef.current.width = rect.width;
    canvasRef.current.height = rect.height;

    const scale = containerRef.current.offsetWidth / rect.width;

    const sx = rect.x;
    const sy = rect.y;
    const sWidth = rect.width;
    const sHeight = rect.height;
    const dx = PADDING / scale;
    const dy = PADDING / scale;
    const dWidth = rect.width - (PADDING * 2) / scale;
    const dHeight = rect.height - (PADDING * 2) / scale;

    ctx.drawImage(canvas, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);

    if (maskColor) {
      ctx.fillStyle = maskColor;
      ctx.globalCompositeOperation = 'source-in';
      ctx.fillRect(0, 0, rect.width, rect.height);
    }
  }, [adapter.transformer, adapter.transformer.nodeRect, adapter.transformer.pixelRect, cache, maskColor]);

  return (
    <Flex
      position="relative"
      ref={containerRef}
      alignItems="center"
      justifyContent="center"
      w={12}
      h={12}
      borderRadius="sm"
      borderWidth={1}
      bg="base.900"
    >
      <Box
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        bgImage={TRANSPARENCY_CHECKER_PATTERN}
        bgSize="5px"
        opacity={0.1}
      />
      <ChakraCanvas ref={canvasRef} objectFit="contain" maxW="full" maxH="full" />
    </Flex>
  );
});

CanvasEntityPreviewImage.displayName = 'CanvasEntityPreviewImage';
