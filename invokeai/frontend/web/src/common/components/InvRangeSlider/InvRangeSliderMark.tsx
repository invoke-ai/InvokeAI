import { RangeSliderMark as ChakraRangeSliderMark } from '@chakra-ui/react';
import type { InvRangeSliderMarkProps } from 'common/components/InvRangeSlider/types';
import { sliderMarkAnimationConstants } from 'common/components/InvSlider/InvSliderMark';
import { motion } from 'framer-motion';
import { memo } from 'react';

export const InvRangeSliderMark = memo(
  ({ value, label, index, total }: InvRangeSliderMarkProps) => {
    if (index === 0) {
      return (
        <ChakraRangeSliderMark
          as={motion.div}
          initial={sliderMarkAnimationConstants.initialFirstLast}
          animate={sliderMarkAnimationConstants.animateFirstLast}
          exit={sliderMarkAnimationConstants.exitFirstLast}
          key={value}
          value={value}
          sx={sliderMarkAnimationConstants.firstMarkStyle}
        >
          {label}
        </ChakraRangeSliderMark>
      );
    }

    if (index === total - 1) {
      return (
        <ChakraRangeSliderMark
          as={motion.div}
          initial={sliderMarkAnimationConstants.initialFirstLast}
          animate={sliderMarkAnimationConstants.animateFirstLast}
          exit={sliderMarkAnimationConstants.exitFirstLast}
          key={value}
          value={value}
          sx={sliderMarkAnimationConstants.lastMarkStyle}
        >
          {label}
        </ChakraRangeSliderMark>
      );
    }

    return (
      <ChakraRangeSliderMark
        as={motion.div}
        initial={sliderMarkAnimationConstants.initialOther}
        animate={sliderMarkAnimationConstants.animateOther}
        exit={sliderMarkAnimationConstants.exitOther}
        key={value}
        value={value}
      >
        {label}
      </ChakraRangeSliderMark>
    );
  }
);

InvRangeSliderMark.displayName = 'InvRangeSliderMark';
