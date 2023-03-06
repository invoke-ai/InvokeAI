import { ChakraProps, SlideDirection } from '@chakra-ui/react';
import { AnimationProps } from 'framer-motion';
import { Enable } from 're-resizable';
import React from 'react';
import { LangDirection } from './types';

export type GetHandleEnablesOptions = {
  direction: SlideDirection;
  langDirection: LangDirection;
};

// Determine handles to enable, taking into account language direction
export const getHandleEnables = ({
  direction,
  langDirection,
}: GetHandleEnablesOptions) => {
  const top = direction === 'bottom';

  const right =
    (langDirection !== 'rtl' && direction === 'left') ||
    (langDirection === 'rtl' && direction === 'right');

  const bottom = direction === 'top';

  const left =
    (langDirection !== 'rtl' && direction === 'right') ||
    (langDirection === 'rtl' && direction === 'left');

  return { top, right, bottom, left };
};

export type GetDefaultSizeOptions = {
  initialWidth?: string | number;
  initialHeight?: string | number;
  direction: SlideDirection;
};

// Get default sizes based on direction and initial values
export const getDefaultSize = ({
  initialWidth,
  initialHeight,
  direction,
}: GetDefaultSizeOptions) => {
  const width =
    initialWidth ?? (['left', 'right'].includes(direction) ? 500 : '100vw');

  const height =
    initialHeight ?? (['top', 'bottom'].includes(direction) ? 500 : '100vh');

  return { width, height };
};

export type GetMinMaxDimensionsOptions = {
  direction: SlideDirection;
  minWidth?: string | number;
  maxWidth?: string | number;
  minHeight?: string | number;
  maxHeight?: string | number;
};

// Get the min/max width/height based on direction and provided values
export const getMinMaxDimensions = ({
  direction,
  minWidth,
  maxWidth,
  minHeight,
  maxHeight,
}: GetMinMaxDimensionsOptions) => {
  const minW =
    minWidth ?? (['left', 'right'].includes(direction) ? 10 : undefined);

  const maxW =
    maxWidth ?? (['left', 'right'].includes(direction) ? '95vw' : undefined);

  const minH =
    minHeight ?? (['top', 'bottom'].includes(direction) ? 10 : undefined);

  const maxH =
    maxHeight ?? (['top', 'bottom'].includes(direction) ? '95vh' : undefined);

  return { minWidth: minW, maxWidth: maxW, minHeight: minH, maxHeight: maxH };
};

export type GetHandleStylesOptions = {
  handleEnables: Enable;
  handleStyle?: React.CSSProperties;
};

// Get handle styles, the enables already have language direction factored in so
// that does not need to be handled here
export const getHandleStyles = ({
  handleEnables,
  handleStyle,
}: GetHandleStylesOptions) => {
  if (!handleStyle) {
    return {};
  }

  const top = handleEnables.top ? handleStyle : {};
  const right = handleEnables.right ? handleStyle : {};
  const bottom = handleEnables.bottom ? handleStyle : {};
  const left = handleEnables.left ? handleStyle : {};

  return {
    top,
    right,
    bottom,
    left,
  };
};

export type GetAnimationsOptions = {
  direction: SlideDirection;
  langDirection: LangDirection;
};

// Get the framer-motion animation props, taking into account language direction
export const getAnimations = ({
  direction,
  langDirection,
}: GetAnimationsOptions): AnimationProps => {
  const baseAnimation = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    // chakra consumes the transition prop, which, for it, is a string.
    // however we know the transition prop will make it to framer motion,
    // which wants it as an object. cast as string to satisfy TS.
    transition: { duration: 0.2, ease: 'easeInOut' },
  };

  const langDirectionFactor = langDirection === 'rtl' ? -1 : 1;

  if (direction === 'top') {
    return {
      ...baseAnimation,
      initial: { y: -999 },
      animate: { y: 0 },
      exit: { y: -999 },
    };
  }

  if (direction === 'right') {
    return {
      ...baseAnimation,
      initial: { x: 999 * langDirectionFactor },
      animate: { x: 0 },
      exit: { x: 999 * langDirectionFactor },
    };
  }

  if (direction === 'bottom') {
    return {
      ...baseAnimation,
      initial: { y: 999 },
      animate: { y: 0 },
      exit: { y: 999 },
    };
  }

  if (direction === 'left') {
    return {
      ...baseAnimation,
      initial: { x: -999 * langDirectionFactor },
      animate: { x: 0 },
      exit: { x: -999 * langDirectionFactor },
    };
  }

  return {};
};

export type GetResizableStylesProps = {
  sx: ChakraProps['sx'];
  direction: SlideDirection;
  handleWidth: number;
  isPinned: boolean;
};

export const getResizableStyles = ({
  isPinned, // TODO add borderRadius for pinned?
  sx,
  direction,
  handleWidth,
}: GetResizableStylesProps): ChakraProps['sx'] => {
  if (isPinned) {
    return sx;
  }

  if (direction === 'top') {
    return {
      borderBottomWidth: handleWidth,
      ...sx,
    };
  }

  if (direction === 'right') {
    return { borderInlineStartWidth: handleWidth, ...sx };
  }

  if (direction === 'bottom') {
    return {
      borderTopWidth: handleWidth,
      ...sx,
    };
  }

  if (direction === 'left') {
    return { borderInlineEndWidth: handleWidth, ...sx };
  }
};
