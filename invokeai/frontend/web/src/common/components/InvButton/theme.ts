import type { StyleFunctionProps } from '@chakra-ui/react';
import { defineStyle, defineStyleConfig } from '@chakra-ui/react';
import { buttonVariantPromptOverlay } from 'features/parameters/components/Prompts/theme';

type Variant = `solid` | `outline` | `ghost` | `link`;

const getBorders = (
  props: StyleFunctionProps,
  variant: Variant
): { borderWidth: string; borderStyle: string } => {
  if (variant !== 'outline') {
    return {
      borderWidth: '0px',
      borderStyle: 'none',
    };
  }
  if (props.size === 'lg') {
    return {
      borderWidth: '1px',
      borderStyle: 'solid',
    };
  }
  if (props.size === 'md') {
    return {
      borderWidth: '1px',
      borderStyle: 'solid',
    };
  }
  return {
    borderWidth: '1px',
    borderStyle: 'solid',
  };
};

const getColors = (
  props: StyleFunctionProps,
  variant: Variant
): {
  bg: string;
  bgDisabled: string;
  bgHover: string;
  fg: string;
  fgDisabled: string;
  fgHover: string;
  borderColor: string;
  borderColorDisabled: string;
  borderColorHover: string;
} => {
  const { colorScheme: c } = props;

  const bgBase = 'base.400';
  const bgColor = c === 'invokeYellow' ? `${c}.500` : `${c}.400`;
  const bgBaseHover = 'base.300';
  const bgColorHover = c === 'invokeYellow' ? `${c}.300` : `${c}.300`;

  const notSolidFg = {
    fg: c === 'base' ? 'base.100' : bgColor,
    fgHover: c === 'base' ? 'base.50' : bgColorHover,
    fgDisabled: 'base.600',
  };
  const noBg = { bg: 'none', bgHover: 'none', bgDisabled: 'none' };
  const noBorder = {
    borderColor: 'none',
    borderColorDisabled: 'none',
    borderColorHover: 'none',
  };

  if (variant === 'ghost') {
    return {
      ...notSolidFg,
      ...noBg,
      ...noBorder,
      bgHover: 'baseAlpha.200',
    };
  }

  if (variant === 'link') {
    return {
      ...notSolidFg,
      ...noBg,
      ...noBorder,
    };
  }
  if (variant === 'outline') {
    return {
      ...notSolidFg,
      ...noBg,
      borderColor: c === 'invokeYellow' ? `${c}Alpha.500` : `${c}Alpha.400`,
      borderColorDisabled: 'base.600',
      borderColorHover:
        c === 'invokeYellow' ? `${c}Alpha.700` : `${c}Alpha.600`,
    };
  }

  // solid
  return {
    bg: c === 'base' ? bgBase : bgColor,
    bgHover: c === 'base' ? bgBaseHover : bgColorHover,
    bgDisabled: 'base.600',
    fg: 'base.900',
    fgHover: 'base.900',
    fgDisabled: 'base.750',
    ...noBorder,
  };
};

const getStyles = (
  props: StyleFunctionProps,
  variant: 'ghost' | 'solid' | 'outline' | 'link'
) => {
  const { borderWidth, borderStyle } = getBorders(props, variant);
  const {
    bg,
    bgDisabled,
    bgHover,
    fg,
    fgDisabled,
    fgHover,
    borderColor,
    borderColorDisabled,
    borderColorHover,
  } = getColors(props, variant);

  const _disabled = {
    bg: bgDisabled,
    color: fgDisabled,
    opacity: 1,
    borderColor: borderColorDisabled,
    svg: {
      fill: fgDisabled,
    },
    _hover: {
      bg: bgDisabled,
      color: fgDisabled,
      borderColor: borderColorDisabled,
      svg: {
        fill: fgDisabled,
      },
    },
  };

  const _base = {
    bg: bg,
    color: fg,
    borderWidth: borderWidth,
    borderStyle: borderStyle,
    borderColor: borderColor,
    svg: {
      fill: fg,
    },
  };

  return {
    ..._base,
    _hover: {
      bg: bgHover,
      color: fgHover,
      borderColor: borderColorHover,
      svg: {
        fill: fgHover,
      },
      _disabled,
    },
    _active: { ..._base },
    _disabled,
  } as const;
};

export const buttonTheme = defineStyleConfig({
  baseStyle: {
    fontWeight: 'bold',
    svg: {
      transitionProperty: 'all',
      transitionDuration: 'faster',
    },
  },
  sizes: {
    sm: {
      fontSize: 'sm',
      px: 3,
      py: 2,
    },
    md: {
      fontSize: 'md',
      px: 4,
      py: 2,
    },
    lg: {
      fontSize: 'lg',
      px: 4,
      py: 2,
      h: 12,
    },
  },
  variants: {
    solid: defineStyle((props) => getStyles(props, 'solid')),
    appTab: defineStyle((_props) => {
      // Has no disabled, invalid, loading, etc state
      return {
        bg: 'none',
        svg: {
          fill: 'base.600',
        },
        _hover: {
          bg: 'none',
          svg: {
            fill: 'base.500',
          },
        },
        '&[data-selected="true"]': {
          bg: 'none',
          svg: {
            fill: 'base.100',
          },
        },
      };
    }),
    outline: defineStyle((props) => getStyles(props, 'outline')),
    ghost: defineStyle((props) => getStyles(props, 'ghost')),
    link: defineStyle((props) => getStyles(props, 'link')),
    promptOverlay: buttonVariantPromptOverlay,
  },
  defaultProps: {
    variant: 'solid',
    colorScheme: 'base',
    size: 'md',
  },
});
