import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';
import { recipes as chakraRecipes, slotRecipes as chakraSlotRecipes } from '@chakra-ui/react/theme';

/** Extend the tooltip recipe; replacing it drops arrow size/background variables. */
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
      // Chakra's `fast` scale-fade drags on an annotation this small.
      _open: { ...chakraSlotRecipes.tooltip.base?.content?._open, animationDuration: 'faster' },
      _closed: { ...chakraSlotRecipes.tooltip.base?.content?._closed, animationDuration: 'faster' },
    },
    arrowTip: {
      ...chakraSlotRecipes.tooltip.base?.arrowTip,
      borderColor: 'border.emphasized',
    },
  },
});

/** Extend the hover-card recipe to preserve arrow variables. Content owns padding; callers must not duplicate it. */
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

/** Extend the popover recipe to preserve arrow size/background variables. */
export const popoverSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.popover,
  base: {
    ...chakraSlotRecipes.popover.base,
    content: {
      ...chakraSlotRecipes.popover.base?.content,
      '--popover-bg': 'colors.bg.muted',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
      _open: { animationStyle: 'slide-fade-in', animationDuration: 'faster' },
      _closed: { animationStyle: 'slide-fade-out', animationDuration: 'faster' },
    },
    arrowTip: {
      ...chakraSlotRecipes.popover.base?.arrowTip,
      borderColor: 'border.emphasized',
    },
  },
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
          roundedTop: 'sm',
          _hover: {
            '&:not([data-selected])': { bg: 'gray.hoverTint/10', color: 'fg' },
          },
        },
      },
      subtle: {
        ...chakraSlotRecipes.tabs.variants?.variant?.subtle,
        trigger: {
          ...chakraSlotRecipes.tabs.variants?.variant?.subtle?.trigger,
          // Translucent hover stays visible on muted chrome; preserve selected accent fills for navigation.
          borderRadius: 'control',
          _hover: {
            '&:not([data-selected])': { bg: 'gray.hoverTint/10', color: 'fg' },
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
  base: {
    ...chakraRecipes.button.base,
    borderRadius: 'control',
    // Chakra's `moderate` hover fade reads as lag on a busy workbench.
    transitionDuration: 'faster',
  },
  variants: {
    ...chakraRecipes.button.variants,
    // Align xs button height with segment tabs.
    size: {
      ...chakraRecipes.button.variants?.size,
      xs: { ...chakraRecipes.button.variants?.size?.xs, h: '7', minW: '7' },
      sm: { ...chakraRecipes.button.variants?.size?.sm, h: '8', minW: '8', px: '3', textStyle: 'xs' },
      md: { ...chakraRecipes.button.variants?.size?.md, h: '9', minW: '9', textStyle: 'xs' },
    },
    variant: {
      ...chakraRecipes.button.variants?.variant,
      // Use translucent hover tint because solid subtle fills disappear on matching surfaces.
      ghost: {
        ...chakraRecipes.button.variants?.variant?.ghost,
        _hover: { bg: 'colorPalette.hoverTint/10' },
        _expanded: { bg: 'colorPalette.hoverTint/10' },
      },
      outline: {
        ...chakraRecipes.button.variants?.variant?.outline,
        _hover: { bg: 'colorPalette.hoverTint/10' },
        _expanded: { bg: 'colorPalette.hoverTint/10' },
      },
      // Give plain buttons the same visible hover fill as ghost buttons.
      plain: {
        ...chakraRecipes.button.variants?.variant?.plain,
        _hover: { bg: 'colorPalette.hoverTint/10' },
      },
      // Use translucent tint so subtle buttons retain their palette hue across surfaces.
      subtle: {
        ...chakraRecipes.button.variants?.variant?.subtle,
        bg: 'colorPalette.hoverTint/22',
        _hover: { bg: 'colorPalette.hoverTint/32' },
        _expanded: { bg: 'colorPalette.hoverTint/32' },
      },
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
      // Use accent selection because neutral fills disappear on lighter section surfaces.
      '--segment-indicator-bg': 'colors.accent.solid',
      '--segment-indicator-shadow': 'none',
      bg: 'transparent',
      borderColor: 'border.subtle',
      // Inner radius subtracts the 1px border from the shared outer radius.
      borderRadius: 'control',
      borderWidth: '1px',
      boxShadow: 'none',
    },
    item: {
      ...chakraSlotRecipes.segmentGroup.base?.item,
      color: 'fg.muted',
      // Use positive z-order; axe can treat negative indicators as beneath the page and measure the wrong
      // contrast.
      zIndex: 1,
      fontWeight: '500',
      transitionDuration: 'faster',
      transitionProperty: 'background, color',
      _before: { display: 'none' },
      _checked: { color: 'accent.contrast' },
      _hover: {
        '&:not([data-state=checked])': { color: 'fg' },
      },
      '&[data-state=checked][data-ssr]': {
        bg: 'accent.solid',
        shadow: 'none',
      },
    },
    indicator: {
      ...chakraSlotRecipes.segmentGroup.base?.indicator,
      // Bind Zag's inline transition variable to the motion-aware duration token.
      '--transition-duration': '{durations.fast}',
      shadow: 'none',
      zIndex: 0,
    },
  },
  variants: {
    ...chakraSlotRecipes.segmentGroup.variants,
    // Subtract root borders from same-size button heights so segment controls align with neighboring buttons.
    size: {
      ...chakraSlotRecipes.segmentGroup.variants?.size,
      // Chakra has no 2xs segment group; derive it from xs styles.
      '2xs': {
        item: {
          ...chakraSlotRecipes.segmentGroup.variants?.size?.xs?.item,
          height: 'calc({sizes.6} - 2px)',
          px: '2',
          textStyle: 'xs',
        },
      },
      xs: {
        item: {
          ...chakraSlotRecipes.segmentGroup.variants?.size?.xs?.item,
          height: 'calc({sizes.7} - 2px)',
          px: '2.5',
        },
      },
      sm: {
        item: {
          ...chakraSlotRecipes.segmentGroup.variants?.size?.sm?.item,
          height: 'calc({sizes.8} - 2px)',
          px: '3.5',
          textStyle: 'xs',
        },
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
    // Hover must not repaint an invalid border neutral; tint instead so hover stays visible.
    _invalid: { borderColor: 'border.error', bg: 'bg.error/60' },
    _expanded: formControlFocused,
    _focusVisible: formControlFocused,
  },
};

const formControlOpen = { borderColor: 'accent.solid' };

/** Use focus-within for composite fields such as InputShell. */
export const inputShellInteraction = {
  ...formControlInteraction,
  _focusWithin: formControlFocused,
  _hover: { ...formControlInteraction._hover, _focusWithin: formControlFocused },
};

/** Scrubber borders follow keyboard/editor focus; pointer clicks use drag state instead. */
export const scrubberInteraction = {
  ...formControlInteraction,
  '&:has(:focus-visible), &[data-editing]': formControlFocused,
  _hover: {
    ...formControlInteraction._hover,
    '&:has(:focus-visible), &[data-editing]': formControlFocused,
  },
};

export const inputRecipe = defineRecipe({
  ...chakraRecipes.input,
  variants: {
    ...chakraRecipes.input.variants,
    // Keep same-named input, select, combobox, and button sizes aligned.
    size: {
      ...chakraRecipes.input.variants?.size,
      xs: { ...chakraRecipes.input.variants?.size?.xs, '--input-height': 'sizes.7' },
      sm: { ...chakraRecipes.input.variants?.size?.sm, '--input-height': 'sizes.8' },
      md: { ...chakraRecipes.input.variants?.size?.md, '--input-height': 'sizes.9' },
    },
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
    size: {
      ...chakraSlotRecipes.numberInput.variants?.size,
      xs: {
        ...chakraSlotRecipes.numberInput.variants?.size?.xs,
        input: { ...chakraSlotRecipes.numberInput.variants?.size?.xs?.input, '--input-height': 'sizes.7' },
      },
      sm: {
        ...chakraSlotRecipes.numberInput.variants?.size?.sm,
        input: { ...chakraSlotRecipes.numberInput.variants?.size?.sm?.input, '--input-height': 'sizes.8' },
      },
      md: {
        ...chakraSlotRecipes.numberInput.variants?.size?.md,
        input: { ...chakraSlotRecipes.numberInput.variants?.size?.md?.input, '--input-height': 'sizes.9' },
      },
    },
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
  borderRadius: 'l2',
  // Use data-danger for destructive menu items.
  '&[data-danger]': {
    color: 'fg.error',
    _highlighted: { bg: 'bg.error' },
    _hover: { bg: 'bg.error' },
  },
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
  lineHeight: 'shorter',
  py: '1',
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
  // Override outline _expanded at variant level; base styles lose to it. Preserve Chakra's variant-map inference
  // in the cast.
  variants: {
    ...chakraSlotRecipes.select.variants,
    size: {
      ...chakraSlotRecipes.select.variants?.size,
      xs: {
        ...chakraSlotRecipes.select.variants?.size?.xs,
        root: { ...chakraSlotRecipes.select.variants?.size?.xs?.root, '--select-trigger-height': 'sizes.7' },
      },
      sm: {
        ...chakraSlotRecipes.select.variants?.size?.sm,
        root: { ...chakraSlotRecipes.select.variants?.size?.sm?.root, '--select-trigger-height': 'sizes.8' },
      },
      md: {
        ...chakraSlotRecipes.select.variants?.size?.md,
        root: { ...chakraSlotRecipes.select.variants?.size?.md?.root, '--select-trigger-height': 'sizes.9' },
      },
    },
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
  } as unknown as typeof chakraSlotRecipes.select.variants,
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
    size: {
      ...chakraSlotRecipes.combobox.variants?.size,
      xs: {
        ...chakraSlotRecipes.combobox.variants?.size?.xs,
        root: { ...chakraSlotRecipes.combobox.variants?.size?.xs?.root, '--combobox-input-height': 'sizes.7' },
      },
      sm: {
        ...chakraSlotRecipes.combobox.variants?.size?.sm,
        root: { ...chakraSlotRecipes.combobox.variants?.size?.sm?.root, '--combobox-input-height': 'sizes.8' },
      },
      md: {
        ...chakraSlotRecipes.combobox.variants?.size?.md,
        root: { ...chakraSlotRecipes.combobox.variants?.size?.md?.root, '--combobox-input-height': 'sizes.9' },
      },
    },
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
      bg: 'bg.subtle',
      borderColor: 'border.subtle',
      borderWidth: '1px',
      color: 'fg',
    },
    header: {
      ...chakraSlotRecipes.dialog.base?.header,
      px: '4',
      pt: '3',
      pb: '2',
    },
    body: {
      ...chakraSlotRecipes.dialog.base?.body,
      px: '4',
      pt: '1.5',
      pb: '4',
    },
    footer: {
      ...chakraSlotRecipes.dialog.base?.footer,
      gap: '2',
      px: '4',
      pt: '1',
      pb: '3',
    },
    title: {
      ...chakraSlotRecipes.dialog.base?.title,
      fontWeight: '700',
      textStyle: 'xs',
    },
    description: {
      ...chakraSlotRecipes.dialog.base?.description,
      color: 'fg.subtle',
      textStyle: 'xs',
    },
    closeTrigger: {
      ...chakraSlotRecipes.dialog.base?.closeTrigger,
      top: '1.5',
      insetEnd: '1.5',
    },
  },
  defaultVariants: {
    ...chakraSlotRecipes.dialog.defaultVariants,
    placement: 'center',
  },
});

/** Check each scrollbar's own overflow axis; Chakra's combined guard leaves phantom thumbs on the other axis. */
export const scrollAreaSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.scrollArea,
  base: {
    ...chakraSlotRecipes.scrollArea.base,
    scrollbar: {
      ...chakraSlotRecipes.scrollArea.base?.scrollbar,
      '&[data-orientation="vertical"]:not([data-overflow-y])': { display: 'none' },
      '&[data-orientation="horizontal"]:not([data-overflow-x])': { display: 'none' },
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
  variants: {
    ...chakraSlotRecipes.slider.variants,
    size: {
      // Fine pointers use smaller thumbs. Update marker center with thumb size and zero marker inset because Zag
      // already applies half-thumb offsets.
      lg: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.lg?.root,
          '--slider-marker-inset': '0px',
          '@media (pointer: fine)': { '--slider-marker-center': '4px', '--slider-thumb-size': 'sizes.3.5' },
        },
      },
      md: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.md?.root,
          '--slider-marker-inset': '0px',
          '@media (pointer: fine)': { '--slider-marker-center': '4px', '--slider-thumb-size': 'sizes.3' },
        },
      },
      sm: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.sm?.root,
          '--slider-marker-inset': '0px',
          '@media (pointer: fine)': { '--slider-marker-center': '3px', '--slider-thumb-size': 'sizes.2.5' },
        },
      },
    },
  } as unknown as typeof chakraSlotRecipes.slider.variants,
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
      '3xs': {
        circle: {
          '--size': '14px',
          '--thickness': '2px',
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

/**
 * Use a linear sweep to avoid endpoint pauses; system.ts disables skeleton motion and gradients under reduced
 * motion.
 */
export const skeletonRecipe = defineRecipe({
  ...chakraRecipes.skeleton,
  variants: {
    ...chakraRecipes.skeleton.variants,
    variant: {
      ...chakraRecipes.skeleton.variants?.variant,
      shine: {
        ...chakraRecipes.skeleton.variants?.variant?.shine,
        '--duration': '2s',
        animation: 'bg-position var(--duration) linear infinite',
        '--end-color': 'colors.bg.emphasized',
        '--start-color': 'color-mix(in oklab, {colors.fg} 8%, {colors.bg.emphasized})',
      },
    },
  } as unknown as typeof chakraRecipes.skeleton.variants,
  defaultVariants: {
    ...chakraRecipes.skeleton.defaultVariants,
    variant: 'shine',
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
    display: 'flex',
    gap: '2',
    textAlign: 'start',
    transition: 'background var(--wb-motion-duration-fast) ease, color var(--wb-motion-duration-fast) ease',
    w: 'full',
    // Keep hover below selected emphasis so pointing does not resemble selection.
    _hover: { bg: 'bg.muted/60' },
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
      selected: { bg: 'bg.emphasized/60', _hover: { bg: 'bg.emphasized/60' } },
      emphasized: { bg: 'bg.emphasized', _hover: { bg: 'bg.emphasized' } },
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
    description: { color: 'fg.muted', fontSize: '2xs', lineHeight: '1.3' },
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

/** Separate variable-height metadata rows with centered dividers; keep padding aligned with the row gap. */
export const dataListSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.dataList,
  base: {
    ...chakraSlotRecipes.dataList.base,
    item: {
      ...chakraSlotRecipes.dataList.base?.item,
      '&:not(:first-child)': {
        // Use borderColor plus top width; side-specific color does not resolve the semantic token.
        borderColor: 'border.subtle',
        borderTopWidth: '1px',
        paddingTop: '1.5',
      },
    },
  },
});
