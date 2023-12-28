import { Flex, Icon } from '@chakra-ui/react';
import { useSize } from '@chakra-ui/react-use-size';
import { AnimatePresence, motion } from 'framer-motion';
import { useRef } from 'react';
import { FaImage } from 'react-icons/fa';

import {
  BOX_SIZE_CSS_CALC,
  ICON_CONTAINER_STYLES,
  MOTION_ICON_ANIMATE,
  MOTION_ICON_EXIT,
  MOTION_ICON_INITIAL,
} from './constants';
import { useAspectRatioPreviewState } from './hooks';
import type { AspectRatioPreviewProps } from './types';

export const AspectRatioPreview = (props: AspectRatioPreviewProps) => {
  const { width: _width, height: _height, icon = FaImage } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const containerSize = useSize(containerRef);

  const { width, height, shouldShowIcon } = useAspectRatioPreviewState({
    width: _width,
    height: _height,
    containerSize,
  });

  return (
    <Flex
      w="full"
      h="full"
      alignItems="center"
      justifyContent="center"
      ref={containerRef}
    >
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
              <Icon as={icon} color="base.700" boxSize={BOX_SIZE_CSS_CALC} />
            </Flex>
          )}
        </AnimatePresence>
      </Flex>
    </Flex>
  );
};
