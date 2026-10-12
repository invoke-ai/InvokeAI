import type { SystemStyleObject } from '@chakra-ui/react';

import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';

import { recipes as chakraRecipes, slotRecipes as chakraSlotRecipes } from './rebase';
import { CONTROL_HEIGHT_PX, type ControlSize } from './scale';

const CONTROL_SIZES = Object.keys(CONTROL_HEIGHT_PX) as ControlSize[];

type SlotStyles = Partial<Record<string, SystemStyleObject>>;

/** The renamed stock step each control size takes padding and text from; heights always come from the scale. */
const CONTROL_BASIS: Record<ControlSize, string> = {
  xs: 'md',
  sm: 'md',
  md: 'md',
  lg: 'lg',
  xl: 'xl',
  '2xl': 'xl',
  '3xl': '3xl',
};

/**
 * Builds every control size from the stock steps, so same-named sizes align across control recipes. A recipe without
 * the basis step (select has no stock 3xl) takes its xl.
 */
const controlSizes = <T>(
  stock: Record<string, T> | undefined,
  build: (styles: T | undefined, height: string, size: ControlSize) => T,
  basis: Record<ControlSize, string> = CONTROL_BASIS
) =>
  Object.fromEntries(
    CONTROL_SIZES.map((size) => [size, build(stock?.[basis[size]] ?? stock?.xl, `{sizes.control.${size}}`, size)])
  );

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
    positioner: {
      ...chakraSlotRecipes.tooltip.base?.positioner,
      // Zag translates the positioner by `--x`/`--y`, which it sets inline a frame after opening. Unset, they leave it
      // at the page's origin, where a tooltip closed within that frame (focus, then a click that disables the trigger)
      // would fade out. Default off-screen, as Zag does for a positioner with no placement.
      '--x': '0px',
      '--y': '-100vh',
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

/**
 * Keep white status text at AA: Chakra dims descriptions to 80%, and its green/orange 600 fills are too light
 * for white at any opacity, so success and warning use the 700 step.
 */
export const toastSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.toast,
  base: {
    ...chakraSlotRecipes.toast.base,
    root: {
      ...chakraSlotRecipes.toast.base?.root,
      // Chakra's `*.contrast` text is black on these fills in dark color modes; the 700 steps need white. Its
      // lightening trigger hover drops white text below AA on red, so status toasts darken on hover instead.
      '&[data-type=error]': {
        ...chakraSlotRecipes.toast.base?.root?.['&[data-type=error]'],
        '--toast-trigger-bg': '{black/20}',
      },
      '&[data-type=success]': {
        ...chakraSlotRecipes.toast.base?.root?.['&[data-type=success]'],
        bg: 'green.700',
        color: 'white',
        '--toast-trigger-bg': '{black/20}',
      },
      '&[data-type=warning]': {
        ...chakraSlotRecipes.toast.base?.root?.['&[data-type=warning]'],
        bg: 'orange.700',
        color: 'white',
        '--toast-trigger-bg': '{black/20}',
      },
    },
    description: { ...chakraSlotRecipes.toast.base?.description, opacity: 1 },
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
      lg: {
        root: {
          '--tabs-height': '{sizes.control.lg}',
          '--tabs-content-padding': 'spacing.2.5',
        },
        trigger: { px: '2.5', py: '0.5', textStyle: 'md' },
      },
      xl: {
        ...chakraSlotRecipes.tabs.variants?.size?.xl,
        trigger: { ...chakraSlotRecipes.tabs.variants?.size?.xl?.trigger, textStyle: 'md' },
      },
      '2xl': {
        ...chakraSlotRecipes.tabs.variants?.size?.['2xl'],
        trigger: { ...chakraSlotRecipes.tabs.variants?.size?.['2xl']?.trigger, textStyle: 'md' },
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
            '&:not([data-selected])': { bg: 'bg.hover' },
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

/** Buttons keep their stock 2xs styles (renamed sm) for the two smallest steps. */
const BUTTON_BASIS: Record<ControlSize, string> = { ...CONTROL_BASIS, xs: 'sm', sm: 'sm' };

const BUTTON_SIZE_TWEAKS: Partial<Record<ControlSize, SystemStyleObject>> = {
  xs: { px: '1.5', _icon: { height: '3', width: '3' } },
  lg: { px: '3', textStyle: 'md' },
  xl: { textStyle: 'md' },
};

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
    // Heights come from the shared scale, so a button aligns with same-sized inputs, selects, and segment tabs.
    size: controlSizes<SystemStyleObject>(
      chakraRecipes.button.variants?.size,
      (styles, height, size) => ({ ...styles, h: height, minW: height, ...BUTTON_SIZE_TWEAKS[size] }),
      BUTTON_BASIS
    ),
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

/**
 * Stock segment groups skip xl; the larger steps take the renamed stock 2xl and 3xl. Those two now subtract the root
 * border like every other step (stock did not), so they are 2px shorter than stock but align with same-sized buttons.
 */
const SEGMENT_BASIS: Record<ControlSize, string> = { ...CONTROL_BASIS, xl: 'lg', '2xl': '2xl' };

const SEGMENT_SIZE_TWEAKS: Partial<Record<ControlSize, SystemStyleObject>> = {
  xs: { px: '1.5' },
  sm: { px: '2' },
  md: { px: '2.5' },
  lg: { px: '3.5', textStyle: 'md' },
};

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
    size: controlSizes<SlotStyles>(
      chakraSlotRecipes.segmentGroup.variants?.size,
      (styles, height, size) => ({
        item: { ...styles?.item, height: `calc(${height} - 2px)`, ...SEGMENT_SIZE_TWEAKS[size] },
      }),
      SEGMENT_BASIS
    ),
  } as unknown as typeof chakraSlotRecipes.segmentGroup.variants,
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

/** A mode-tinted shell (e.g. semantic search) keeps its tint on hover; focus and errors still win. */
export const warningInputShellInteraction = {
  ...inputShellInteraction,
  _hover: { ...inputShellInteraction._hover, borderColor: 'fg.warning' },
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
    size: controlSizes<SystemStyleObject>(chakraRecipes.input.variants?.size, (styles, height) => ({
      ...styles,
      '--input-height': height,
    })),
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
    size: controlSizes<SlotStyles>(chakraSlotRecipes.numberInput.variants?.size, (styles, height) => ({
      ...styles,
      input: { ...styles?.input, '--input-height': height },
    })),
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
  _highlighted: { bg: 'bg.hover' },
  _hover: { bg: 'bg.hover' },
  _focusVisible: {
    outline: '2px solid',
    outlineColor: 'accent.solid',
    outlineOffset: '-2px',
  },
};

export const dropdownGroupLabel = {
  color: 'fg.subtle',
  fontSize: 'xs',
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
});

export const selectSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.select,
  // Override outline _expanded at variant level; base styles lose to it. Preserve Chakra's variant-map inference
  // in the cast.
  variants: {
    ...chakraSlotRecipes.select.variants,
    size: controlSizes<SlotStyles>(chakraSlotRecipes.select.variants?.size, (styles, height) => ({
      ...styles,
      root: { ...styles?.root, '--select-trigger-height': height },
    })),
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
    size: controlSizes<SlotStyles>(chakraSlotRecipes.combobox.variants?.size, (styles, height) => ({
      ...styles,
      root: { ...styles?.root, '--combobox-input-height': height },
    })),
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

/**
 * Dialogs stack from the modal token, below the popover token menus and popovers use. zag copies a menu's z-index to its
 * positioner once, before the layer index lands, so a menu opened in a second, stacked dialog sat under that dialog.
 */
const DIALOG_LAYER = { '--dialog-z-index': 'zIndex.modal' } as const;

export const dialogSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.dialog,
  base: {
    ...chakraSlotRecipes.dialog.base,
    backdrop: { ...chakraSlotRecipes.dialog.base?.backdrop, ...DIALOG_LAYER },
    positioner: { ...chakraSlotRecipes.dialog.base?.positioner, ...DIALOG_LAYER },
    content: {
      ...chakraSlotRecipes.dialog.base?.content,
      ...DIALOG_LAYER,
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
      textStyle: 'md',
    },
    description: {
      ...chakraSlotRecipes.dialog.base?.description,
      color: 'fg.subtle',
      textStyle: 'md',
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
      // Above sticky content (z-index 1) so opaque pinned headers never cover the thumb.
      zIndex: 2,
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
      xl: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.xl?.root,
          '--slider-marker-inset': '0px',
          '@media (pointer: fine)': { '--slider-marker-center': '4px', '--slider-thumb-size': 'sizes.3.5' },
        },
      },
      lg: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.lg?.root,
          '--slider-marker-inset': '0px',
          '@media (pointer: fine)': { '--slider-marker-center': '4px', '--slider-thumb-size': 'sizes.3' },
        },
      },
      md: {
        root: {
          ...chakraSlotRecipes.slider.variants?.size?.md?.root,
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
      sm: {
        circle: {
          '--size': '16px',
          '--thickness': '3px',
        },
        valueText: {
          textStyle: 'xs',
        },
      },
      xs: {
        circle: {
          '--size': '14px',
          '--thickness': '2px',
        },
        valueText: {
          textStyle: 'xs',
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
      textStyle: 'xs',
    },
    transparencyGrid: {
      ...chakraSlotRecipes.colorPicker.base?.transparencyGrid,
      borderRadius: 'inherit',
    },
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

const rowFocusRing = {
  outline: '2px solid',
  outlineColor: 'accent.solid',
  outlineOffset: '-2px',
} as const;

/** Hover, extended to a row whose context menu is open so the row it acts on stays marked. */
const ROW_POINTED = '&:is(:hover, [data-hover], [data-menu-open]):not(:disabled, [data-disabled], [data-static])';

/** One row surface for every list-like control; `Row` and `ListItem` both build on it. */
const rowSurface = {
  borderRadius: 'sm',
  textAlign: 'start',
  transition: 'background var(--wb-motion-duration-fast) ease, color var(--wb-motion-duration-fast) ease',
  w: 'full',
  // Keep the pointed fill below selected emphasis so pointing does not resemble selection.
  [ROW_POINTED]: { bg: 'bg.hover' },
  _disabled: { cursor: 'not-allowed', opacity: 0.5 },
} as const;

/** Emphasis levels: `accent` marks the one active row, `selected` marks checked rows. */
const rowTones = {
  none: {},
  muted: { bg: 'bg.muted' },
  selected: { bg: 'bg.emphasized/60', [ROW_POINTED]: { bg: 'bg.emphasized/60' } },
  emphasized: { bg: 'bg.emphasized', [ROW_POINTED]: { bg: 'bg.emphasized' } },
  brand: {
    bg: 'brand.subtle',
    color: 'brand.fg',
    [ROW_POINTED]: { bg: 'brand.subtle' },
  },
  accent: {
    bg: 'accent.solid',
    color: 'accent.contrast',
    [ROW_POINTED]: { bg: 'accent.solid' },
  },
} as const;

export const rowRecipe = defineRecipe({
  base: {
    ...rowSurface,
    alignItems: 'center',
    display: 'flex',
    gap: '2',
    _focusVisible: rowFocusRing,
  },
  variants: {
    active: rowTones,
  },
  defaultVariants: { active: 'none' },
});

/**
 * A list row: an optional checkbox beside one primary button, never inside it. The ring wraps the whole row when
 * the button has focus; the accent tone recolors muted text so it stays legible on the solid fill.
 */
export const listItemSlotRecipe = defineSlotRecipe({
  slots: [
    'root',
    'check',
    'primary',
    'body',
    'titleLine',
    'title',
    'badges',
    'description',
    'trailing',
    'actions',
    'detail',
  ],
  base: {
    root: {
      ...rowSurface,
      alignItems: 'stretch',
      display: 'flex',
      // Lets `detail` take its own line under the row.
      flexWrap: 'wrap',
      minW: 0,
      position: 'relative',
      '&:has([data-list-primary]:focus-visible)': rowFocusRing,
      // Rows of a page being replaced stay readable; only the pointer says the list is working.
      '&[data-busy]': { cursor: 'progress' },
    },
    check: {
      alignItems: 'center',
      display: 'flex',
      flexShrink: 0,
      ps: '2',
    },
    primary: {
      alignItems: 'center',
      appearance: 'none',
      bg: 'transparent',
      border: 0,
      borderRadius: 'inherit',
      color: 'inherit',
      display: 'flex',
      flex: 1,
      font: 'inherit',
      minW: 0,
      outline: 'none',
      textAlign: 'start',
    },
    body: {
      display: 'flex',
      flex: 1,
      flexDirection: 'column',
      gap: '0.5',
      minW: 0,
    },
    titleLine: {
      alignItems: 'center',
      display: 'flex',
      gap: '1.5',
      minW: 0,
    },
    title: {
      fontSize: 'md',
      fontWeight: '600',
      lineHeight: 'shorter',
    },
    badges: {
      alignItems: 'center',
      display: 'flex',
      flexShrink: 0,
      gap: '1',
    },
    description: {
      color: 'fg.muted',
      fontSize: 'xs',
      lineHeight: 'shorter',
      minW: 0,
    },
    trailing: {
      alignItems: 'center',
      color: 'fg.muted',
      display: 'flex',
      flexShrink: 0,
      fontSize: 'xs',
      gap: '1.5',
    },
    // Controls beside the primary button, never inside it.
    actions: {
      alignItems: 'center',
      display: 'flex',
      flexShrink: 0,
      gap: '0.5',
      pe: '1',
    },
    // Row-owned content on its own line, inset to the primary button's text.
    detail: {
      flexBasis: '100%',
      minW: 0,
      pb: '2',
      px: '2',
    },
  },
  variants: {
    active: {
      none: {},
      selected: { root: rowTones.selected },
      accent: {
        root: rowTones.accent,
        description: { color: 'accent.contrast', opacity: 0.85 },
        trailing: { color: 'accent.contrast' },
      },
    },
    density: {
      compact: {
        root: { minH: '7' },
        primary: { gap: '2', px: '2', py: '1' },
      },
      regular: {
        root: { minH: '10' },
        primary: { gap: '2', px: '2', py: '1.5' },
      },
      comfortable: {
        root: { minH: '13' },
        primary: { gap: '2.5', px: '2', py: '1.5' },
      },
      // Comfortable's media one step tighter, for rows whose detail line already adds height.
      snug: {
        root: { minH: '12' },
        primary: { gap: '2.5', px: '1.5', py: '1.5' },
        detail: { pb: '1.5', px: '1.5' },
      },
    },
  },
  defaultVariants: { active: 'none', density: 'regular' },
});

/** Section labels inside lists; the count sits beside the label in a quieter tone. */
export const listSectionHeaderSlotRecipe = defineSlotRecipe({
  slots: ['root', 'label', 'count'],
  base: {
    root: {
      alignItems: 'flex-end',
      display: 'flex',
      gap: '1.5',
      h: '8',
      minW: 0,
      pb: '1.5',
      ps: '2',
    },
    label: {
      color: 'fg.muted',
      fontSize: 'xs',
      fontWeight: '700',
      letterSpacing: '0.04em',
      lineHeight: 'shorter',
      minW: 0,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      textTransform: 'uppercase',
      whiteSpace: 'nowrap',
    },
    count: {
      color: 'fg.muted',
      flexShrink: 0,
      fontSize: 'xs',
      fontVariantNumeric: 'tabular-nums',
      lineHeight: 'shorter',
    },
  },
});

export const chipRecipe = defineRecipe({
  base: {
    alignItems: 'center',
    borderRadius: 'sm',
    display: 'inline-flex',
    flexShrink: '0',
    fontSize: 'xs',
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
    fontSize: 'xs',
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
    name: { color: 'fg', fontSize: 'lg', fontWeight: '600' },
    description: { color: 'fg.muted', fontSize: 'xs', lineHeight: '1.3' },
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

const resizeGripIdle = { bg: 'border.emphasized' } as const;
const resizeGripActive = { bg: 'fg.subtle' } as const;
const resizeGripFocused = { bg: 'accent.solid' } as const;
const resizeGrip = {
  ...resizeGripIdle,
  borderRadius: 'full',
  content: '""',
  position: 'absolute',
  transition: 'background var(--wb-motion-duration-fast) ease',
} as const;

// A capsule on the divider line; odd sizes keep its edges on whole pixels either side of the 1px line.
const resizeCapsule = {
  bg: 'bg',
  borderColor: 'border.emphasized',
  borderRadius: 'full',
  borderWidth: '1px',
  boxShadow: 'xs',
  content: '""',
  position: 'absolute',
  transition: 'border-color var(--wb-motion-duration-fast) ease',
} as const;
const resizeCapsuleStates = {
  '&:hover::after, &[data-dragging]::after': { borderColor: 'fg.subtle' },
  '&:focus-visible::after': { outline: '2px solid {colors.accent.solid}', outlineOffset: '1px' },
  '&[data-collapse-armed]::after': { opacity: 0 },
  // A region outline runs along the hidden line and behind the capsule, which takes its colour.
  '[data-line-hidden] > &::after': { borderColor: 'accent.solid' },
} as const;

/**
 * Every resize affordance: a hairline divider with a capsule grip at its middle. The hit strip covers the line and
 * extends toward the end side, away from the scrollbar of the pane before it.
 */
export const resizeHandleSlotRecipe = defineSlotRecipe({
  slots: ['root', 'handle'],
  base: {
    root: {
      flexShrink: '0',
      position: 'relative',
      // Above region outlines (zIndex 4) so the capsule masks an outline passing through it.
      zIndex: 5,
      // The line is a border, not a 1px background: at fractional display scales a background can round to two
      // device pixels while every other border draws one. A pseudo-element keeps the hit strip's offsets unshifted.
      _before: { borderColor: 'border.subtle', content: '""', inset: '0', position: 'absolute' },
      '&[data-line-hidden]::before, &:has([data-collapse-armed])::before': { borderColor: 'transparent' },
    },
    handle: {
      outline: 'none',
      position: 'absolute',
      touchAction: 'none',
    },
  },
  variants: {
    orientation: {
      vertical: {
        root: { alignSelf: 'stretch', w: '1px', _before: { borderLeftWidth: '1px' } },
        handle: {
          bottom: '0',
          cursor: 'col-resize',
          left: '-1px',
          top: '0',
          w: '9px',
          _after: { ...resizeCapsule, h: '25px', left: '-3px', top: '50%', transform: 'translateY(-50%)', w: '9px' },
          ...resizeCapsuleStates,
        },
      },
      horizontal: {
        root: { alignSelf: 'stretch', h: '1px', _before: { borderTopWidth: '1px' } },
        handle: {
          cursor: 'row-resize',
          h: '9px',
          left: '0',
          right: '0',
          top: '-1px',
          _after: { ...resizeCapsule, h: '9px', left: '50%', top: '-3px', transform: 'translateX(-50%)', w: '25px' },
          ...resizeCapsuleStates,
        },
      },
      // The grip bent into an L along the window's bottom-right corner.
      corner: {
        root: {},
        handle: {
          bottom: '0',
          cursor: 'nwse-resize',
          h: '4',
          right: '0',
          w: '4',
          _before: { ...resizeGrip, bottom: '3px', h: '3px', right: '3px', w: '3' },
          _after: { ...resizeGrip, bottom: '3px', h: '3', right: '3px', w: '3px' },
          '&:hover::before, &:hover::after, &[data-dragging]::before, &[data-dragging]::after': resizeGripActive,
          '&:focus-visible::before, &:focus-visible::after': resizeGripFocused,
        },
      },
    },
  },
});
