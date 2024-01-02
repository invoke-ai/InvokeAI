import type { MotionProps } from 'framer-motion';

/**
 * Props for the animation of the MenuList.
 */
export const menuListMotionProps: MotionProps = {
  variants: {
    enter: {
      visibility: 'visible',
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.07,
        ease: [0.4, 0, 0.2, 1],
      },
    },
    exit: {
      transitionEnd: {
        visibility: 'hidden',
      },
      opacity: 0,
      scale: 0.8,
      transition: {
        duration: 0.07,
        easings: 'easeOut',
      },
    },
  },
};
