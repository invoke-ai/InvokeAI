import { Box, Flex, Icon, Image } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { memo, useCallback, useRef } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

const INITIAL_POS = '50%';
const HANDLE_WIDTH = 2;
const HANDLE_WIDTH_PX = `${HANDLE_WIDTH}px`;
const HANDLE_HITBOX = 20;
const HANDLE_HITBOX_PX = `${HANDLE_HITBOX}px`;
const HANDLE_LEFT_INITIAL_PX = `calc(${INITIAL_POS} - ${HANDLE_HITBOX / 2}px)`;
const HANDLE_INNER_LEFT_INITIAL_PX = `${HANDLE_HITBOX / 2 - HANDLE_WIDTH / 2}px`;

export const ImageSliderComparison = memo(() => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageAContainerRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<HTMLDivElement>(null);

  const updateHandlePos = useCallback((clientX: number) => {
    if (!containerRef.current || !imageAContainerRef.current || !handleRef.current) {
      return;
    }
    const { x, width } = containerRef.current.getBoundingClientRect();
    const rawHandlePos = ((clientX - x) * 100) / width;
    const handleWidthPct = (HANDLE_WIDTH * 100) / width;
    const newHandlePos = Math.min(100 - handleWidthPct, Math.max(0, rawHandlePos));
    imageAContainerRef.current.style.width = `${newHandlePos}%`;
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

  const { imageA, imageB } = useAppSelector((s) => {
    const images = s.gallery.selection.slice(-2);
    return { imageA: images[0] ?? null, imageB: images[1] ?? null };
  });

  if (imageA && !imageB) {
    return <CurrentImagePreview />;
  }

  if (!imageA || !imageB) {
    return null;
  }

  return (
    <Flex
      w="full"
      h="full"
      maxW="full"
      maxH="full"
      position="relative"
      border="1px solid cyan"
      alignItems="center"
      justifyContent="center"
    >
      <Flex
        ref={containerRef}
        w={imageA.width}
        h={imageA.height}
        maxW="full"
        maxH="full"
        position="relative"
        userSelect="none"
        border="1px solid magenta"
        objectFit="contain"
        overflow="hidden"
      >
        <Image src={imageB.image_url} objectFit="contain" objectPosition="top left" />
        <Flex
          ref={imageAContainerRef}
          w={INITIAL_POS}
          h={imageA.height}
          position="absolute"
          overflow="hidden"
          objectFit="contain"
        >
          <Image
            src={imageA.image_url}
            width={`${imageA.width}px`}
            height={`${imageA.height}px`}
            maxW="full"
            maxH="full"
            objectFit="none"
            objectPosition="left"
          />
        </Flex>
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
      </Flex>
      <Box
        position="absolute"
        top={0}
        right={0}
        bottom={0}
        left={0}
        id="image-comparison-overlay"
        onMouseDown={onMouseDown}
        userSelect="none"
      />
    </Flex>
  );
});

ImageSliderComparison.displayName = 'ImageSliderComparison';
