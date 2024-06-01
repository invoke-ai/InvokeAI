import { Box, Flex, Icon, Image, Text } from '@invoke-ai/ui-library';
import { useMeasure } from '@reactuses/core';
import { useAppSelector } from 'app/store/storeHooks';
import { preventDefault } from 'common/util/stopPropagation';
import type { Dimensions } from 'features/canvas/store/canvasTypes';
import { STAGE_BG_DATAURL } from 'features/controlLayers/util/renderers';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

const DROP_SHADOW = 'drop-shadow(0px 0px 4px rgb(0, 0, 0))';
const INITIAL_POS = '50%';
const HANDLE_WIDTH = 1;
const HANDLE_WIDTH_PX = `${HANDLE_WIDTH}px`;
const HANDLE_HITBOX = 20;
const HANDLE_HITBOX_PX = `${HANDLE_HITBOX}px`;
const HANDLE_INNER_LEFT_PX = `${HANDLE_HITBOX / 2 - HANDLE_WIDTH / 2}px`;
const HANDLE_LEFT_INITIAL_PX = `calc(${INITIAL_POS} - ${HANDLE_HITBOX / 2}px)`;

type Props = {
  /**
   * The first image to compare
   */
  firstImage: ImageDTO;
  /**
   * The second image to compare
   */
  secondImage: ImageDTO;
};

export const ImageComparisonSlider = memo(({ firstImage, secondImage }: Props) => {
  const { t } = useTranslation();
  const sliderFit = useAppSelector((s) => s.gallery.sliderFit);
  // How far the handle is from the left - this will be a CSS calculation that takes into account the handle width
  const [left, setLeft] = useState(HANDLE_LEFT_INITIAL_PX);
  // How wide the first image is
  const [width, setWidth] = useState(INITIAL_POS);
  const handleRef = useRef<HTMLDivElement>(null);
  // If the container size is not provided, use an internal ref and measure - can cause flicker on mount tho
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize] = useMeasure(containerRef);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  // To keep things smooth, we use RAF to update the handle position & gate it to 60fps
  const rafRef = useRef<number | null>(null);
  const lastMoveTimeRef = useRef<number>(0);

  const fittedSize = useMemo<Dimensions>(() => {
    // Fit the first image to the container
    if (containerSize.width === 0 || containerSize.height === 0) {
      return { width: firstImage.width, height: firstImage.height };
    }
    const targetAspectRatio = containerSize.width / containerSize.height;
    const imageAspectRatio = firstImage.width / firstImage.height;

    let width: number;
    let height: number;

    if (firstImage.width <= containerSize.width && firstImage.height <= containerSize.height) {
      return { width: firstImage.width, height: firstImage.height };
    }

    if (imageAspectRatio > targetAspectRatio) {
      // Image is wider than container's aspect ratio
      width = containerSize.width;
      height = width / imageAspectRatio;
    } else {
      // Image is taller than container's aspect ratio
      height = containerSize.height;
      width = height * imageAspectRatio;
    }
    return { width, height };
  }, [containerSize, firstImage.height, firstImage.width]);

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
    <Flex
      ref={containerRef}
      w="full"
      h="full"
      maxW="full"
      maxH="full"
      position="relative"
      alignItems="center"
      justifyContent="center"
    >
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
          w={fittedSize.width}
          h={fittedSize.height}
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
            backgroundImage={STAGE_BG_DATAURL}
            backgroundRepeat="repeat"
            opacity={0.2}
          />
          <Image
            position="relative"
            id="image-comparison-second-image"
            src={secondImage.image_url}
            fallbackSrc={secondImage.thumbnail_url}
            w={sliderFit === 'fill' ? fittedSize.width : (fittedSize.width * secondImage.width) / firstImage.width}
            h={sliderFit === 'fill' ? fittedSize.height : (fittedSize.height * secondImage.height) / firstImage.height}
            maxW={fittedSize.width}
            maxH={fittedSize.height}
            objectFit={sliderFit}
            objectPosition="top left"
          />
          <Text
            position="absolute"
            bottom={4}
            insetInlineEnd={4}
            textOverflow="clip"
            whiteSpace="nowrap"
            filter={DROP_SHADOW}
            color="base.50"
          >
            {t('gallery.compareImage')}
          </Text>
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
              w={fittedSize.width}
              h={fittedSize.height}
              objectFit="cover"
              objectPosition="top left"
            />
            <Text
              position="absolute"
              bottom={4}
              insetInlineStart={4}
              textOverflow="clip"
              whiteSpace="nowrap"
              filter={DROP_SHADOW}
              color="base.50"
            >
              {t('gallery.viewerImage')}
            </Text>
          </Box>
          <Flex
            id="image-comparison-handle"
            ref={handleRef}
            position="absolute"
            top={0}
            bottom={0}
            left={left}
            w={HANDLE_HITBOX_PX}
            tabIndex={-1}
            cursor="ew-resize"
            filter={DROP_SHADOW}
          >
            <Box
              id="image-comparison-handle-divider"
              w={HANDLE_WIDTH_PX}
              h="full"
              bg="base.50"
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
