import { useSize } from '@chakra-ui/react-use-size';
import { Flex, Icon } from '@invoke-ai/ui-library';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { AnimatePresence, motion } from 'framer-motion';
import { useMemo, useRef } from 'react';
import { PiFrameCorners } from 'react-icons/pi';

import {
  BOX_SIZE_CSS_CALC,
  ICON_CONTAINER_STYLES,
  ICON_HIGH_CUTOFF,
  ICON_LOW_CUTOFF,
  MOTION_ICON_ANIMATE,
  MOTION_ICON_EXIT,
  MOTION_ICON_INITIAL,
} from './constants';

export const AspectRatioPreview = () => {
  const ctx = useImageSizeContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const containerSize = useSize(containerRef);

  const shouldShowIcon = useMemo(
    () => ctx.aspectRatioState.value < ICON_HIGH_CUTOFF && ctx.aspectRatioState.value > ICON_LOW_CUTOFF,
    [ctx.aspectRatioState.value]
  );

  const { width, height } = useMemo(() => {
    if (!containerSize) {
      return { width: 0, height: 0 };
    }

    let width = ctx.width;
    let height = ctx.height;

    if (ctx.width > ctx.height) {
      width = containerSize.width;
      height = width / ctx.aspectRatioState.value;
    } else {
      height = containerSize.height;
      width = height * ctx.aspectRatioState.value;
    }

    return { width, height };
  }, [containerSize, ctx.width, ctx.height, ctx.aspectRatioState.value]);

  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center" ref={containerRef}>
      <Flex
        bg="blackAlpha.400"
        borderRadius="base"
        width={`${width}px`}
        height={`${height}px`}
        alignItems="center"
        justifyContent="center"
      >
        <AnimatePresence>
          {shouldShowIcon && (
            <Flex
              as={motion.div}
              initial={MOTION_ICON_INITIAL}
              animate={MOTION_ICON_ANIMATE}
              exit={MOTION_ICON_EXIT}
              style={ICON_CONTAINER_STYLES}
            >
              <Icon as={PiFrameCorners} color="base.700" boxSize={BOX_SIZE_CSS_CALC} />
            </Flex>
          )}
        </AnimatePresence>
      </Flex>
    </Flex>
  );
};
