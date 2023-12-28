import { SliderMark as ChakraSliderMark } from '@chakra-ui/react';
import type { SystemStyleObject } from '@chakra-ui/styled-system';
import type { InvSliderMarkProps } from 'common/components/InvSlider/types';
import type { MotionProps } from 'framer-motion';
import { motion } from 'framer-motion';

const initialFirstLast: MotionProps['initial'] = { opacity: 0, y: 10 };
const initialOther = { ...initialFirstLast, x: '-50%' };

const animateFirstLast: MotionProps['animate'] = {
  opacity: 1,
  y: 0,
  transition: { duration: 0.2, ease: 'easeOut' },
};
const animateOther = { ...animateFirstLast, x: '-50%' };

const exitFirstLast: MotionProps['exit'] = {
  opacity: 0,
  y: 10,
  transition: { duration: 0.2, ease: 'anticipate' },
};
const exitOther = { ...exitFirstLast, x: '-50%' };

const firstMarkStyle: SystemStyleObject = {
  insetInlineStart: '0 !important',
  insetInlineEnd: 'unset !important',
};
const lastMarkStyle: SystemStyleObject = {
  insetInlineStart: 'unset !important',
  insetInlineEnd: '0 !important',
};

export const sliderMarkAnimationConstants = {
  initialFirstLast,
  initialOther,
  exitFirstLast,
  exitOther,
  animateFirstLast,
  animateOther,
  firstMarkStyle,
  lastMarkStyle,
};

export const InvSliderMark = ({
  value,
  label,
  index,
  total,
}: InvSliderMarkProps) => {
  if (index === 0) {
    return (
      <ChakraSliderMark
        as={motion.div}
        initial={sliderMarkAnimationConstants.initialFirstLast}
        animate={sliderMarkAnimationConstants.animateFirstLast}
        exit={sliderMarkAnimationConstants.exitFirstLast}
        key={value}
        value={value}
        sx={sliderMarkAnimationConstants.firstMarkStyle}
      >
        {label}
      </ChakraSliderMark>
    );
  }

  if (index === total - 1) {
    return (
      <ChakraSliderMark
        as={motion.div}
        initial={sliderMarkAnimationConstants.initialFirstLast}
        animate={sliderMarkAnimationConstants.animateFirstLast}
        exit={sliderMarkAnimationConstants.exitFirstLast}
        key={value}
        value={value}
        sx={sliderMarkAnimationConstants.lastMarkStyle}
      >
        {label}
      </ChakraSliderMark>
    );
  }

  return (
    <ChakraSliderMark
      as={motion.div}
      initial={sliderMarkAnimationConstants.initialOther}
      animate={sliderMarkAnimationConstants.animateOther}
      exit={sliderMarkAnimationConstants.exitOther}
      key={value}
      value={value}
    >
      {label}
    </ChakraSliderMark>
  );
};
