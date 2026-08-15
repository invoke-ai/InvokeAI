import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';
import { recipes as chakraRecipes, slotRecipes as chakraSlotRecipes } from '@chakra-ui/react/theme';

/**
 * Tooltip chrome: raised surface with a hairline stroke instead of inverted
 * fill. Extends Chakra's default recipe — replacing it wholesale would drop
 * the `arrow` slot's `--arrow-size`/`--arrow-background` vars, which renders
 * arrows at zero size (invisible).
 */
export const tooltipSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.tooltip,
  base: {
    ...chakraSlotRecipes.tooltip.base,
    content: {
      ...chakraSlotRecipes.tooltip.base?.content,
      '--tooltip-bg': 'colors.bg.muted',
      bg: 'var(--tooltip-bg)',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
    },
    arrowTip: {
      ...chakraSlotRecipes.tooltip.base?.arrowTip,
      borderColor: 'border.emphasized',
    },
  },
});

/**
 * Feature hint cards. Same raised surface as the tooltip so the two read as one
 * family, one step wider for prose. Extends Chakra's default recipe: the arrow
 * slots derive `--arrow-background` from `--hovercard-bg`, so replacing the base
 * wholesale would render the arrow unfilled.
 *
 * The content owns its padding — cards must not add their own, or the two stack.
 * Chakra's `md` default (20px) reads as a dialog rather than an annotation, so
 * these default to `xs`.
 */
export const hoverCardSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.hoverCard,
  base: {
    ...chakraSlotRecipes.hoverCard.base,
    content: {
      ...chakraSlotRecipes.hoverCard.base?.content,
      '--hovercard-bg': 'colors.bg.muted',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
      maxWidth: '18rem',
    },
    arrowTip: {
      ...chakraSlotRecipes.hoverCard.base?.arrowTip,
      borderColor: 'border.emphasized',
    },
  },
  defaultVariants: { size: 'xs' },
});

export const tabsSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.tabs,
  base: {
    ...chakraSlotRecipes.tabs.base,
    trigger: {
      ...chakraSlotRecipes.tabs.base?.trigger,
      transitionDuration: 'faster',
      transitionProperty: 'background, border-color, color',
    },
  },
  variants: {
    ...chakraSlotRecipes.tabs.variants,
    size: {
      ...chakraSlotRecipes.tabs.variants?.size,
      xs: {
        root: {
          '--tabs-height': 'sizes.8',
          '--tabs-content-padding': 'spacing.2.5',
        },
        trigger: { px: '2.5', py: '0.5', textStyle: 'xs' },
      },
      sm: {
        ...chakraSlotRecipes.tabs.variants?.size?.sm,
        trigger: { ...chakraSlotRecipes.tabs.variants?.size?.sm?.trigger, textStyle: 'xs' },
      },
      md: {
        ...chakraSlotRecipes.tabs.variants?.size?.md,
        trigger: { ...chakraSlotRecipes.tabs.variants?.size?.md?.trigger, textStyle: 'xs' },
      },
    },
    variant: {
      ...chakraSlotRecipes.tabs.variants?.variant,
      line: {
        ...chakraSlotRecipes.tabs.variants?.variant?.line,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.line?.trigger,
          _hover: {
            '&:not([data-selected])': { bg: 'bg.muted/60', color: 'fg' },
          },
        },
      },
      subtle: {
        ...chakraSlotRecipes.tabs.variants?.variant?.subtle,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.subtle?.trigger,
          _hover: {
            '&:not([data-selected])': { bg: 'bg.muted' },
          },
        },
      },
      enclosed: {
        ...chakraSlotRecipes.tabs.variants?.variant?.enclosed,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.enclosed?.trigger,
          _hover: {
            '&:not([data-selected])': { bg: 'bg.emphasized' },
          },
        },
      },
      outline: {
        ...chakraSlotRecipes.tabs.variants?.variant?.outline,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.outline?.trigger,
          _hover: {
            '&:not([data-selected])': {
              bg: 'bg.muted',
              borderColor: 'border.emphasized',
            },
          },
        },
      },
      plain: {
        ...chakraSlotRecipes.tabs.variants?.variant?.plain,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.plain?.trigger,
          _hover: {
            '&:not([data-selected])': { bg: 'bg.muted/40', color: 'fg' },
          },
        },
      },
    },
  } as unknown as typeof chakraSlotRecipes.tabs.variants,
});

export const buttonRecipe = defineRecipe({
  ...chakraRecipes.button,
  variants: {
    ...chakraRecipes.button.variants,
    size: {
      ...chakraRecipes.button.variants?.size,
      sm: { ...chakraRecipes.button.variants?.size?.sm, textStyle: 'xs' },
      md: { ...chakraRecipes.button.variants?.size?.md, textStyle: 'xs' },
    },
  } as unknown as typeof chakraRecipes.button.variants,
});

export const segmentGroupSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.segmentGroup,
  base: {
    ...chakraSlotRecipes.segmentGroup.base,
    root: {
      ...chakraSlotRecipes.segmentGroup.base?.root,
      '--segment-radius': 'radii.sm',
      '--segment-indicator-bg': 'colors.bg.emphasized',
      '--segment-indicator-shadow': 'none',
      bg: 'transparent',
      borderColor: 'border.subtle',
      borderRadius: 'md',
      borderWidth: '1px',
      boxShadow: 'none',
    },
    item: {
      ...chakraSlotRecipes.segmentGroup.base?.item,
      color: 'fg.muted',
      fontWeight: '500',
      transitionDuration: 'faster',
      transitionProperty: 'background, color',
      _before: { display: 'none' },
      _checked: { color: 'fg' },
      _hover: {
        '&:not([data-state=checked])': { color: 'fg' },
      },
      '&[data-state=checked][data-ssr]': {
        bg: 'bg.emphasized',
        shadow: 'none',
      },
    },
    indicator: {
      ...chakraSlotRecipes.segmentGroup.base?.indicator,
      shadow: 'none',
    },
  },
  variants: {
    ...chakraSlotRecipes.segmentGroup.variants,
    size: {
      ...chakraSlotRecipes.segmentGroup.variants?.size,
      xs: {
        item: { ...chakraSlotRecipes.segmentGroup.variants?.size?.xs?.item, px: '2.5' },
      },
      sm: {
        item: { ...chakraSlotRecipes.segmentGroup.variants?.size?.sm?.item, px: '3', textStyle: 'xs' },
      },
    },
  } as unknown as typeof chakraSlotRecipes.segmentGroup.variants,
  defaultVariants: {
    ...chakraSlotRecipes.segmentGroup.defaultVariants,
    size: 'xs',
  },
});

const formControlFocused = {
  '--focus-ring-color': 'var(--focus-color) !important',
  borderColor: 'accent.solid',
  boxShadow: 'none !important',
  outline: 'none !important',
  _invalid: {
    '--focus-ring-color': 'var(--chakra-colors-border-error) !important',
    borderColor: 'border.error',
  },
};

const formControlNoFocusRing = {
  focusVisibleRing: undefined,
  _focusVisible: formControlFocused,
} as const;

export const formControlInteraction = {
  '--focus-color': 'var(--chakra-colors-accent-solid)',
  ...formControlNoFocusRing,
  transitionDuration: 'fast',
  transitionProperty: 'border-color, background',
  _focusVisible: formControlFocused,
  _invalid: { borderColor: 'border.error' },
  _hover: {
    borderColor: 'border.emphasized',
    _expanded: formControlFocused,
    _focusVisible: formControlFocused,
  },
};

const formControlOpen = { borderColor: 'accent.solid' };

export const inputRecipe = defineRecipe({
  ...chakraRecipes.input,
  variants: {
    ...chakraRecipes.input.variants,
    variant: {
      ...chakraRecipes.input.variants?.variant,
      outline: { ...chakraRecipes.input.variants?.variant?.outline, ...formControlNoFocusRing },
      subtle: { ...chakraRecipes.input.variants?.variant?.subtle, ...formControlNoFocusRing },
    },
  } as unknown as typeof chakraRecipes.input.variants,
  base: {
    ...chakraRecipes.input.base,
    ...formControlInteraction,
  },
});

export const textareaRecipe = defineRecipe({
  ...chakraRecipes.textarea,
  variants: {
    ...chakraRecipes.textarea.variants,
    variant: {
      ...chakraRecipes.textarea.variants?.variant,
      outline: { ...chakraRecipes.textarea.variants?.variant?.outline, ...formControlNoFocusRing },
      subtle: { ...chakraRecipes.textarea.variants?.variant?.subtle, ...formControlNoFocusRing },
    },
  } as unknown as typeof chakraRecipes.textarea.variants,
  base: {
    ...chakraRecipes.textarea.base,
    ...formControlInteraction,
  },
});

export const numberInputSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.numberInput,
  variants: {
    ...chakraSlotRecipes.numberInput.variants,
    variant: {
      ...chakraSlotRecipes.numberInput.variants?.variant,
      outline: {
        ...chakraSlotRecipes.numberInput.variants?.variant?.outline,
        input: {
          ...chakraSlotRecipes.numberInput.variants?.variant?.outline?.input,
          ...formControlNoFocusRing,
        },
      },
      subtle: {
        ...chakraSlotRecipes.numberInput.variants?.variant?.subtle,
        input: {
          ...chakraSlotRecipes.numberInput.variants?.variant?.subtle?.input,
          ...formControlNoFocusRing,
        },
      },
    },
  } as unknown as typeof chakraSlotRecipes.numberInput.variants,
  base: {
    ...chakraSlotRecipes.numberInput.base,
    input: {
      ...chakraSlotRecipes.numberInput.base?.input,
      ...formControlInteraction,
    },
  },
});

export const dropdownContent = {
  bg: 'bg.muted',
  borderColor: 'border.emphasized',
  borderRadius: 'md',
  borderWidth: '1px',
  boxShadow: 'lg',
  color: 'fg',
};

export const dropdownItem = {
  _highlighted: { bg: 'bg.emphasized' },
  _hover: { bg: 'bg.emphasized' },
  _focusVisible: {
    outline: '2px solid',
    outlineColor: 'accent.solid',
    outlineOffset: '-2px',
  },
};

export const dropdownGroupLabel = {
  color: 'fg.subtle',
  fontSize: '2xs',
  fontWeight: '600',
  letterSpacing: '0.02em',
  textTransform: 'uppercase',
};

export const menuSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.menu,
  base: {
    ...chakraSlotRecipes.menu.base,
    content: {
      ...chakraSlotRecipes.menu.base?.content,
      ...dropdownContent,
    },
    item: {
      ...chakraSlotRecipes.menu.base?.item,
      ...dropdownItem,
    },
    itemGroupLabel: {
      ...chakraSlotRecipes.menu.base?.itemGroupLabel,
      ...dropdownGroupLabel,
    },
    separator: {
      ...chakraSlotRecipes.menu.base?.separator,
      bg: 'border.subtle',
    },
  },
  defaultVariants: {
    ...chakraSlotRecipes.menu.defaultVariants,
    size: 'sm',
  },
});

export const selectSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.select,
  // The outline variant carries its own `_expanded` (border.emphasized) which
  // would override a base-level open style, so the accent open state lives on
  // the variant too. The cast keeps defineSlotRecipe's variant inference
  // anchored to Chakra's own map, which the spread-with-override loses.
  variants: {
    ...chakraSlotRecipes.select.variants,
    variant: {
      ...chakraSlotRecipes.select.variants?.variant,
      outline: {
        ...chakraSlotRecipes.select.variants?.variant?.outline,
        trigger: {
          ...chakraSlotRecipes.select.variants?.variant?.outline?.trigger,
          ...formControlNoFocusRing,
          _expanded: formControlOpen,
        },
      },
      subtle: {
        ...chakraSlotRecipes.select.variants?.variant?.subtle,
        trigger: {
          ...chakraSlotRecipes.select.variants?.variant?.subtle?.trigger,
          ...formControlNoFocusRing,
        },
      },
    },
  } as typeof chakraSlotRecipes.select.variants,
  base: {
    ...chakraSlotRecipes.select.base,
    trigger: {
      ...chakraSlotRecipes.select.base?.trigger,
      ...formControlInteraction,
      _expanded: formControlOpen,
    },
    content: {
      ...chakraSlotRecipes.select.base?.content,
      ...dropdownContent,
    },
    item: {
      ...chakraSlotRecipes.select.base?.item,
      ...dropdownItem,
    },
    itemGroupLabel: {
      ...chakraSlotRecipes.select.base?.itemGroupLabel,
      ...dropdownGroupLabel,
    },
  },
});

export const comboboxSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.combobox,
  variants: {
    ...chakraSlotRecipes.combobox.variants,
    variant: {
      ...chakraSlotRecipes.combobox.variants?.variant,
      outline: {
        ...chakraSlotRecipes.combobox.variants?.variant?.outline,
        input: {
          ...chakraSlotRecipes.combobox.variants?.variant?.outline?.input,
          ...formControlNoFocusRing,
        },
      },
      subtle: {
        ...chakraSlotRecipes.combobox.variants?.variant?.subtle,
        input: {
          ...chakraSlotRecipes.combobox.variants?.variant?.subtle?.input,
          ...formControlNoFocusRing,
        },
      },
    },
  } as unknown as typeof chakraSlotRecipes.combobox.variants,
  base: {
    ...chakraSlotRecipes.combobox.base,
    content: {
      ...chakraSlotRecipes.combobox.base?.content,
      ...dropdownContent,
    },
    input: {
      ...chakraSlotRecipes.combobox.base?.input,
      ...formControlInteraction,
      _expanded: formControlOpen,
    },
    item: {
      ...chakraSlotRecipes.combobox.base?.item,
      ...dropdownItem,
    },
    itemGroupLabel: {
      ...chakraSlotRecipes.combobox.base?.itemGroupLabel,
      ...dropdownGroupLabel,
    },
  },
});

export const dialogSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.dialog,
  base: {
    ...chakraSlotRecipes.dialog.base,
    content: {
      ...chakraSlotRecipes.dialog.base?.content,
      borderColor: 'border.subtle',
      borderWidth: '1px',
    },
  },
});

export const sliderSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.slider,
  base: {
    ...chakraSlotRecipes.slider.base,
    markerLabel: {
      ...chakraSlotRecipes.slider.base?.markerLabel,
      color: 'fg.subtle',
      fontSize: '0.5rem',
      lineHeight: '1',
    },
  },
});

export const progressCircleSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.progressCircle,
  variants: {
    ...chakraSlotRecipes.progressCircle.variants,
    size: {
      ...chakraSlotRecipes.progressCircle.variants?.size,
      '2xs': {
        circle: {
          '--size': '16px',
          '--thickness': '3px',
        },
        valueText: {
          textStyle: '2xs',
        },
      },
    },
  },
});

// Extending the stock recipe preserves the CSS variables that size thumbs and swatches.
export const colorPickerSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.colorPicker,
  base: {
    ...chakraSlotRecipes.colorPicker.base,
    content: {
      ...chakraSlotRecipes.colorPicker.base?.content,
      ...dropdownContent,
      gap: '2',
      p: '2',
      width: '64',
    },
    area: {
      ...chakraSlotRecipes.colorPicker.base?.area,
      height: '140px',
      boxShadow: 'inset 0 0 0 1px {colors.border.subtle}',
    },
    areaThumb: {
      ...chakraSlotRecipes.colorPicker.base?.areaThumb,
      boxShadow: '0 0 0 1px {colors.border.image}',
    },
    channelSliderThumb: {
      ...chakraSlotRecipes.colorPicker.base?.channelSliderThumb,
      boxShadow: '0 0 0 1px {colors.border.image}',
    },
    channelSliderTrack: {
      ...chakraSlotRecipes.colorPicker.base?.channelSliderTrack,
      boxShadow: 'inset 0 0 0 1px {colors.border.subtle}',
    },
    swatch: {
      ...chakraSlotRecipes.colorPicker.base?.swatch,
      borderRadius: 'l1',
      boxShadow: 'inset 0 0 0 1px {colors.border.image}',
    },
    swatchTrigger: {
      ...chakraSlotRecipes.colorPicker.base?.swatchTrigger,
      borderColor: 'transparent',
      borderRadius: 'l1',
      borderWidth: '1px',
      transitionDuration: 'fast',
      transitionProperty: 'border-color',
      _hover: { borderColor: 'border.emphasized' },
      _focusVisible: {
        outline: '2px solid',
        outlineColor: 'accent.solid',
        outlineOffset: '1px',
      },
    },
    channelInput: {
      ...chakraSlotRecipes.colorPicker.base?.channelInput,
      ...formControlInteraction,
      fontVariantNumeric: 'tabular-nums',
      px: '1',
      textAlign: 'center',
    },
    channelText: {
      ...chakraSlotRecipes.colorPicker.base?.channelText,
      color: 'fg.subtle',
      textStyle: '2xs',
    },
    transparencyGrid: {
      ...chakraSlotRecipes.colorPicker.base?.transparencyGrid,
      borderRadius: 'inherit',
    },
  },
  defaultVariants: {
    ...chakraSlotRecipes.colorPicker.defaultVariants,
    size: 'xs',
  },
});

export const panelRecipe = defineRecipe({
  base: {
    bg: 'bg.subtle',
    borderColor: 'border.subtle',
    borderRadius: 'md',
    borderWidth: '1px',
    display: 'flex',
    flexDirection: 'column',
    minH: '0',
    minW: '0',
  },
  variants: {
    tone: {
      surface: {},
      raised: { bg: 'bg.muted' },
      inset: { bg: 'bg.inset' },
      control: { bg: 'bg.emphasized', borderColor: 'transparent' },
    },
    density: {
      none: {},
      sm: { gap: '1.5', p: '2' },
      md: { gap: '2', p: '3' },
    },
  },
  defaultVariants: { tone: 'surface', density: 'none' },
});

export const rowRecipe = defineRecipe({
  base: {
    alignItems: 'center',
    borderRadius: 'sm',
    cursor: 'pointer',
    display: 'flex',
    gap: '2',
    textAlign: 'start',
    transition: 'background var(--wb-motion-duration-fast) ease, color var(--wb-motion-duration-fast) ease',
    w: 'full',
    _hover: { bg: 'bg.emphasized' },
    _focusVisible: {
      outline: '2px solid',
      outlineColor: 'accent.solid',
      outlineOffset: '-2px',
    },
    _disabled: { cursor: 'not-allowed', opacity: 0.5 },
  },
  variants: {
    active: {
      none: {},
      muted: { bg: 'bg.muted' },
      brand: {
        bg: 'brand.subtle',
        color: 'brand.fg',
        _hover: { bg: 'brand.subtle' },
      },
      accent: {
        bg: 'accent.solid',
        color: 'accent.contrast',
        _hover: { bg: 'accent.solid' },
      },
    },
  },
  defaultVariants: { active: 'none' },
});

export const chipRecipe = defineRecipe({
  base: {
    alignItems: 'center',
    borderRadius: 'sm',
    display: 'inline-flex',
    flexShrink: '0',
    fontSize: '2xs',
    fontWeight: '500',
    gap: '1.5',
    px: '2',
    py: '0.5',
    whiteSpace: 'nowrap',
  },
  variants: {
    tone: {
      neutral: {},
      brand: { bg: 'brand.subtle', color: 'brand.fg' },
      accent: { color: 'accent.solid' },
      error: { color: 'fg.error' },
      success: { color: 'fg.success' },
      warning: { color: 'fg.warning' },
    },
  },
  defaultVariants: { tone: 'neutral' },
});

export const fieldLabelRecipe = defineRecipe({
  base: {
    color: 'fg.muted',
    fontSize: '2xs',
    fontWeight: '600',
    letterSpacing: '0.03em',
  },
});

export const themeCardRecipe = defineSlotRecipe({
  slots: ['root', 'preview', 'swatch', 'body', 'name', 'description', 'indicator'],
  base: {
    root: {
      alignItems: 'stretch',
      bg: 'bg.subtle',
      borderColor: 'border.subtle',
      borderRadius: 'lg',
      borderWidth: '1px',
      cursor: 'pointer',
      display: 'flex',
      flexDirection: 'column',
      gap: '2.5',
      overflow: 'hidden',
      p: '3',
      textAlign: 'left',
      transition:
        'border-color var(--wb-motion-duration-fast) ease, background var(--wb-motion-duration-fast) ease, transform var(--wb-motion-duration-fast) ease',
      _hover: { borderColor: 'border.emphasized' },
      _focusVisible: {
        outline: '2px solid',
        outlineColor: 'accent.solid',
        outlineOffset: '2px',
      },
    },
    preview: {
      borderColor: 'border.subtle',
      borderRadius: 'md',
      borderWidth: '1px',
      display: 'flex',
      h: '8',
      overflow: 'hidden',
    },
    swatch: { flex: '1' },
    body: {
      alignItems: 'flex-start',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.5',
    },
    name: { color: 'fg', fontSize: 'sm', fontWeight: '600' },
    description: { color: 'fg.subtle', fontSize: '2xs', lineHeight: '1.3' },
    indicator: {
      alignItems: 'center',
      borderRadius: 'full',
      color: 'accent.solid',
      display: 'flex',
      h: '4',
      justifyContent: 'center',
      opacity: 0,
      w: '4',
    },
  },
  variants: {
    selected: {
      true: {
        root: { borderColor: 'accent.solid', bg: 'bg.muted' },
        indicator: { opacity: 1 },
      },
      false: {},
    },
  },
  defaultVariants: { selected: false },
});
