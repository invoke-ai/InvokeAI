import { SlideDirection } from '@chakra-ui/react';
import { AnimationProps } from 'framer-motion';
import { HandleStyles } from 're-resizable';
import { CSSProperties } from 'react';
import { LangDirection } from './types';

export type GetHandleEnablesOptions = {
  direction: SlideDirection;
  langDirection: LangDirection;
};

/**
 * Determine handles to enable. `re-resizable` doesn't handle RTL, so we have to do that here.
 */
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

  return {
    top,
    right,
    bottom,
    left,
  };
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
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
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

  return {
    ...(minW ? { minWidth: minW } : {}),
    ...(maxW ? { maxWidth: maxW } : {}),
    ...(minH ? { minHeight: minH } : {}),
    ...(maxH ? { maxHeight: maxH } : {}),
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
  isPinned: boolean;
  isResizable: boolean;
  direction: SlideDirection;
};

// Expand the handle hitbox
const HANDLE_INTERACT_PADDING = '0.75rem';

// Visible padding around handle
const HANDLE_PADDING = '1rem';

const HANDLE_WIDTH_PINNED = '2px';
const HANDLE_WIDTH_UNPINNED = '5px';

// Get the styles for the container and handle. Do not need to handle langDirection here bc we use direction-agnostic CSS
export const getStyles = ({
  isPinned,
  isResizable,
  direction,
}: GetResizableStylesProps): {
  containerStyles: CSSProperties; // technically this could be ChakraProps['sx'], but we cannot use this for HandleStyles so leave it as CSSProperties to be consistent
  handleStyles: HandleStyles;
} => {
  if (!isResizable) {
    return { containerStyles: {}, handleStyles: {} };
  }

  const handleWidth = isPinned ? HANDLE_WIDTH_PINNED : HANDLE_WIDTH_UNPINNED;

  // Calculate the positioning offset of the handle hitbox so it is centered over the handle
  const handleOffset = `calc((2 * ${HANDLE_INTERACT_PADDING} + ${handleWidth}) / -2)`;

  if (direction === 'top') {
    return {
      containerStyles: {
        borderBottomWidth: handleWidth,
        paddingBottom: HANDLE_PADDING,
      },
      handleStyles: {
        top: {
          paddingTop: HANDLE_INTERACT_PADDING,
          paddingBottom: HANDLE_INTERACT_PADDING,
          bottom: handleOffset,
        },
      },
    };
  }

  if (direction === 'left') {
    return {
      containerStyles: {
        borderInlineEndWidth: handleWidth,
        paddingInlineEnd: HANDLE_PADDING,
      },
      handleStyles: {
        right: {
          paddingInlineStart: HANDLE_INTERACT_PADDING,
          paddingInlineEnd: HANDLE_INTERACT_PADDING,
          insetInlineEnd: handleOffset,
        },
      },
    };
  }

  if (direction === 'bottom') {
    return {
      containerStyles: {
        borderTopWidth: handleWidth,
        paddingTop: HANDLE_PADDING,
      },
      handleStyles: {
        bottom: {
          paddingTop: HANDLE_INTERACT_PADDING,
          paddingBottom: HANDLE_INTERACT_PADDING,
          top: handleOffset,
        },
      },
    };
  }

  if (direction === 'right') {
    return {
      containerStyles: {
        borderInlineStartWidth: handleWidth,
        paddingInlineStart: HANDLE_PADDING,
      },
      handleStyles: {
        left: {
          paddingInlineStart: HANDLE_INTERACT_PADDING,
          paddingInlineEnd: HANDLE_INTERACT_PADDING,
          insetInlineStart: handleOffset,
        },
      },
    };
  }

  return { containerStyles: {}, handleStyles: {} };
};

// Chakra's Slide does not handle langDirection, so we need to do it here
export const getSlideDirection = (
  direction: SlideDirection,
  langDirection: LangDirection
) => {
  if (['top', 'bottom'].includes(direction)) {
    return direction;
  }

  if (direction === 'left') {
    if (langDirection === 'rtl') {
      return 'right';
    }
    return 'left';
  }

  if (direction === 'right') {
    if (langDirection === 'rtl') {
      return 'left';
    }
    return 'right';
  }

  return 'left';
};

// Hack to correct the width of panels while pinned and unpinned, due to different padding in pinned vs unpinned
export const parseAndPadSize = (size?: number, padding?: number) => {
  if (!size) {
    return undefined;
  }

  if (!padding) {
    return size;
  }

  return size + padding;
};
