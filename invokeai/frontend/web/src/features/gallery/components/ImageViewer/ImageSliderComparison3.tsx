import { Box, Flex, Icon } from '@invoke-ai/ui-library';
import { useMeasure } from '@reactuses/core';
import { memo, useCallback, useMemo, useRef } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const INITIAL_POS = '50%';
const HANDLE_WIDTH = 2;
const HANDLE_WIDTH_PX = `${HANDLE_WIDTH}px`;
const HANDLE_HITBOX = 20;
const HANDLE_HITBOX_PX = `${HANDLE_HITBOX}px`;
const HANDLE_LEFT_INITIAL_PX = `calc(${INITIAL_POS} - ${HANDLE_HITBOX / 2}px)`;
const HANDLE_INNER_LEFT_INITIAL_PX = `${HANDLE_HITBOX / 2 - HANDLE_WIDTH / 2}px`;

type Props = {
  firstImage: ImageDTO;
  secondImage: ImageDTO;
};

export const ImageSliderComparison = memo(({ firstImage, secondImage }: Props) => {
  const secondImageContainerRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize] = useMeasure(containerRef);

  const updateHandlePos = useCallback((clientX: number) => {
    if (!secondImageContainerRef.current || !handleRef.current || !containerRef.current) {
      return;
    }
    const { x, width } = containerRef.current.getBoundingClientRect();
    const rawHandlePos = ((clientX - x) * 100) / width;
    const handleWidthPct = (HANDLE_WIDTH * 100) / width;
    const newHandlePos = Math.min(100 - handleWidthPct, Math.max(0, rawHandlePos));
    secondImageContainerRef.current.style.width = `${newHandlePos}%`;
    handleRef.current.style.left = `calc(${newHandlePos}% - ${HANDLE_HITBOX / 2}px)`;
  }, []);

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      updateHandlePos(e.clientX);
    },
    [updateHandlePos]
  );

  const onMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', onMouseMove);
  }, [onMouseMove]);

  const onMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      updateHandlePos(e.clientX);
      window.addEventListener('mouseup', onMouseUp, { once: true });
      window.addEventListener('mousemove', onMouseMove);
    },
    [onMouseMove, onMouseUp, updateHandlePos]
  );

  const fittedSize = useMemo(() => {
    let width = containerSize.width;
    let height = containerSize.height;
    const aspectRatio = firstImage.width / firstImage.height;
    if (firstImage.width > firstImage.height) {
      width = firstImage.width;
      height = width / aspectRatio;
    } else {
      height = firstImage.height;
      width = height * aspectRatio;
    }
    return { width, height };
  }, [containerSize.height, containerSize.width, firstImage.height, firstImage.width]);

  console.log({ containerSize, fittedSize });

  return (
    <Flex w="full" h="full" maxW="full" maxH="full" position="relative" alignItems="center" justifyContent="center">
      <Flex
        id="image-comparison-container"
        ref={containerRef}
        w="full"
        h="full"
        maxW="full"
        maxH="full"
        position="relative"
        border="1px solid cyan"
        alignItems="center"
        justifyContent="center"
      >
        <Box
          position="relative"
          id="image-comparison-first-image-container"
          w={fittedSize.width}
          h={fittedSize.height}
          maxW="full"
          maxH="full"
          backgroundImage={`url(${secondImage.image_url})`}
          backgroundSize="contain"
          backgroundRepeat="no-repeat"
          userSelect="none"
          border="1px solid green"
          overflow="hidden"
        >
          <Box
            id="image-comparison-second-image-container"
            ref={secondImageContainerRef}
            backgroundImage={`url(${firstImage.image_url})`}
            backgroundSize="auto"
            backgroundPosition="top left"
            backgroundRepeat="no-repeat"
            w={INITIAL_POS}
            h="full"
            maxW="full"
            maxH="full"
            position="absolute"
            top={0}
            left={0}
            right={0}
            bottom={0}
          />
          <Flex
            id="image-comparison-handle"
            ref={handleRef}
            position="absolute"
            top={0}
            bottom={0}
            left={HANDLE_LEFT_INITIAL_PX}
            w={HANDLE_HITBOX_PX}
            tabIndex={-1}
            cursor="ew-resize"
            filter="drop-shadow(0px 0px 4px rgb(0, 0, 0))"
          >
            <Box
              w={HANDLE_WIDTH_PX}
              h="full"
              bg="base.50"
              shadow="dark-lg"
              position="absolute"
              top={0}
              left={HANDLE_INNER_LEFT_INITIAL_PX}
            />
            <Flex gap={4} position="absolute" left="50%" top="50%" transform="translate(-50%, 0)">
              <Icon as={PiCaretLeftBold} />
              <Icon as={PiCaretRightBold} />
            </Flex>
          </Flex>
          <Box
            id="image-comparison-interaction-overlay"
            position="absolute"
            top={0}
            right={0}
            bottom={0}
            left={0}
            onMouseDown={onMouseDown}
            userSelect="none"
            bg="rgba(255,0,0,0.3)"
          />
        </Box>
      </Flex>
    </Flex>
  );
});

ImageSliderComparison.displayName = 'ImageSliderComparison';
