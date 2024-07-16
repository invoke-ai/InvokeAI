import { useSize } from '@chakra-ui/react-use-size';
import { Flex, Icon } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useMemo, useRef } from 'react';
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

export const AspectRatioIconPreview = memo(() => {
  const bbox = useAppSelector((s) => s.canvasV2.bbox);
  const containerRef = useRef<HTMLDivElement>(null);
  const containerSize = useSize(containerRef);

  const shouldShowIcon = useMemo(
    () => bbox.aspectRatio.value < ICON_HIGH_CUTOFF && bbox.aspectRatio.value > ICON_LOW_CUTOFF,
    [bbox.aspectRatio.value]
  );

  const { width, height } = useMemo(() => {
    if (!containerSize) {
      return { width: 0, height: 0 };
    }

    let width = bbox.rect.width;
    let height = bbox.rect.height;

    if (bbox.rect.width > bbox.rect.height) {
      width = containerSize.width;
      height = width / bbox.aspectRatio.value;
    } else {
      height = containerSize.height;
      width = height * bbox.aspectRatio.value;
    }

    return { width, height };
  }, [containerSize, bbox.rect.width, bbox.rect.height, bbox.aspectRatio.value]);

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
});

AspectRatioIconPreview.displayName = 'AspectRatioIconPreview';
