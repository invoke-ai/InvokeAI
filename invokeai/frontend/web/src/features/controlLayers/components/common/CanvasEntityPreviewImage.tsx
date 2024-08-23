import { Box, chakra, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useEntityAdapter } from 'features/controlLayers/contexts/EntityAdapterContext';
import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { memo, useEffect, useRef } from 'react';

const ChakraCanvas = chakra.canvas;

export const CanvasEntityPreviewImage = memo(() => {
  const adapter = useEntityAdapter();
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

    ctx.drawImage(canvas, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
  }, [adapter.transformer, adapter.transformer.nodeRect, adapter.transformer.pixelRect, cache]);

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
