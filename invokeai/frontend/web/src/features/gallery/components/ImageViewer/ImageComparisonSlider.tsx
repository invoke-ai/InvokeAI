import { Box, Flex, Icon, Image } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { preventDefault } from 'common/util/stopPropagation';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import type { Dimensions } from 'features/controlLayers/store/types';
import { ImageComparisonLabel } from 'features/gallery/components/ImageViewer/ImageComparisonLabel';
import { selectComparisonFit } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import type { ComparisonProps } from './common';
import { DROP_SHADOW, fitDimsToContainer, getSecondImageDims } from './common';

const INITIAL_POS = '50%';
const HANDLE_WIDTH = 2;
const HANDLE_WIDTH_PX = `${HANDLE_WIDTH}px`;
const HANDLE_HITBOX = 20;
const HANDLE_HITBOX_PX = `${HANDLE_HITBOX}px`;
const HANDLE_INNER_LEFT_PX = `${HANDLE_HITBOX / 2 - HANDLE_WIDTH / 2}px`;
const HANDLE_LEFT_INITIAL_PX = `calc(${INITIAL_POS} - ${HANDLE_HITBOX / 2}px)`;

export const ImageComparisonSlider = memo(({ firstImage, secondImage, containerDims }: ComparisonProps) => {
  const comparisonFit = useAppSelector(selectComparisonFit);
  // How far the handle is from the left - this will be a CSS calculation that takes into account the handle width
  const [left, setLeft] = useState(HANDLE_LEFT_INITIAL_PX);
  // How wide the first image is
  const [width, setWidth] = useState(INITIAL_POS);
  const handleRef = useRef<HTMLDivElement>(null);
  // To manage aspect ratios, we need to know the size of the container
  const imageContainerRef = useRef<HTMLDivElement>(null);
  // To keep things smooth, we use RAF to update the handle position & gate it to 60fps
  const rafRef = useRef<number | null>(null);
  const lastMoveTimeRef = useRef<number>(0);

  const fittedDims = useMemo<Dimensions>(
    () => fitDimsToContainer(containerDims, firstImage),
    [containerDims, firstImage]
  );

  const compareImageDims = useMemo<Dimensions>(
    () => getSecondImageDims(comparisonFit, fittedDims, firstImage, secondImage),
    [comparisonFit, fittedDims, firstImage, secondImage]
  );

  const updateHandlePos = useCallback((clientX: number) => {
    if (!handleRef.current || !imageContainerRef.current) {
      return;
    }
    lastMoveTimeRef.current = performance.now();
    const { x, width } = imageContainerRef.current.getBoundingClientRect();
    const rawHandlePos = ((clientX - x) * 100) / width;
    const handleWidthPct = (HANDLE_WIDTH * 100) / width;
    const newHandlePos = Math.min(100 - handleWidthPct, Math.max(0, rawHandlePos));
    setWidth(`${newHandlePos}%`);
    setLeft(`calc(${newHandlePos}% - ${HANDLE_HITBOX / 2}px)`);
  }, []);

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (rafRef.current === null && performance.now() > lastMoveTimeRef.current + 1000 / 60) {
        rafRef.current = window.requestAnimationFrame(() => {
          updateHandlePos(e.clientX);
          rafRef.current = null;
        });
      }
    },
    [updateHandlePos]
  );

  const onMouseUp = useCallback(() => {
    window.removeEventListener('mousemove', onMouseMove);
  }, [onMouseMove]);

  const onMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      // Update the handle position immediately on click
      updateHandlePos(e.clientX);
      window.addEventListener('mouseup', onMouseUp, { once: true });
      window.addEventListener('mousemove', onMouseMove);
    },
    [onMouseMove, onMouseUp, updateHandlePos]
  );

  useEffect(
    () => () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    },
    []
  );

  return (
    <Flex w="full" h="full" maxW="full" maxH="full" position="relative" alignItems="center" justifyContent="center">
      <Flex
        id="image-comparison-wrapper"
        w="full"
        h="full"
        maxW="full"
        maxH="full"
        position="absolute"
        alignItems="center"
        justifyContent="center"
      >
        <Box
          ref={imageContainerRef}
          position="relative"
          id="image-comparison-image-container"
          w={fittedDims.width}
          h={fittedDims.height}
          maxW="full"
          maxH="full"
          userSelect="none"
          overflow="hidden"
          borderRadius="base"
        >
          <Box
            id="image-comparison-bg"
            position="absolute"
            top={0}
            left={0}
            right={0}
            bottom={0}
            backgroundImage={TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL}
            backgroundRepeat="repeat"
          />
          <Image
            position="relative"
            id="image-comparison-second-image"
            src={secondImage.image_url}
            fallbackSrc={secondImage.thumbnail_url}
            w={compareImageDims.width}
            h={compareImageDims.height}
            maxW={fittedDims.width}
            maxH={fittedDims.height}
            objectFit={comparisonFit}
            objectPosition="top left"
          />
          <ImageComparisonLabel type="second" />
          <Box
            id="image-comparison-first-image-container"
            position="absolute"
            top={0}
            left={0}
            right={0}
            bottom={0}
            w={width}
            overflow="hidden"
          >
            <Image
              id="image-comparison-first-image"
              src={firstImage.image_url}
              fallbackSrc={firstImage.thumbnail_url}
              w={fittedDims.width}
              h={fittedDims.height}
              objectFit="cover"
              objectPosition="top left"
            />
            <ImageComparisonLabel type="first" />
          </Box>
          <Flex
            id="image-comparison-handle"
            ref={handleRef}
            position="absolute"
            top={0}
            bottom={0}
            left={left}
            w={HANDLE_HITBOX_PX}
            cursor="ew-resize"
            filter={DROP_SHADOW}
            opacity={0.8}
            color="base.50"
          >
            <Box
              id="image-comparison-handle-divider"
              w={HANDLE_WIDTH_PX}
              h="full"
              bg="currentColor"
              shadow="dark-lg"
              position="absolute"
              top={0}
              left={HANDLE_INNER_LEFT_PX}
            />
            <Flex
              id="image-comparison-handle-icons"
              gap={4}
              position="absolute"
              left="50%"
              top="50%"
              transform="translate(-50%, 0)"
              filter={DROP_SHADOW}
            >
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
            onContextMenu={preventDefault}
            userSelect="none"
            cursor="ew-resize"
          />
        </Box>
      </Flex>
    </Flex>
  );
});

ImageComparisonSlider.displayName = 'ImageComparisonSlider';
