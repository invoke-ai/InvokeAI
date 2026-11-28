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

/** A recipe without the basis step (select has no stock 3xl) takes its xl. */
const controlSizes = <T>(
  stock: Record<string, T> | undefined,
  build: (styles: T | undefined, height: string, size: ControlSize) => T,
  basis: Record<ControlSize, string> = CONTROL_BASIS
) =>
  Object.fromEntries(
    CONTROL_SIZES.map((size) => [size, build(stock?.[basis[size]] ?? stock?.xl, `{sizes.control.${size}}`, size)])
  );

/**
 * The arrow box's transparent border (corner radius plus margin) keeps the tip off the content's rounded corners.
 * A zero-size arrow gets no border, so the panel stays where its placement puts it.
 */
const arrowClearOfCorners = (visibleSize: string) => ({
  '--arrow-visible-size': visibleSize,
  '--arrow-clearance': 'min(calc({radii.md} + 2px), calc(var(--arrow-visible-size) * 1000))',
  '--arrow-size': 'calc(var(--arrow-visible-size) + 2 * var(--arrow-clearance))',
  borderColor: 'transparent',
  borderStyle: 'solid',
  borderWidth: 'var(--arrow-clearance)',
  pointerEvents: 'none',
});

/** Zag's offset lands on the padding edge; moving out by the 1px border centres it on the outer edge. */
const arrowInsideContent = {
  zIndex: '-1',
  '[data-placement^=bottom] > &': { translate: '0 -1px' },
  '[data-placement^=top] > &': { translate: '0 1px' },
  '[data-placement^=right] > &': { translate: '-1px 0' },
  '[data-placement^=left] > &': { translate: '1px 0' },
};

/**
 * Clipped to the outer half plus a ~3px band over the content edge, which absorbs fractional-DPR snapping (125%).
 * The clip also bounds hit testing, so the visible tip counts as part of the surface.
 */
const arrowTipOutsideContent = {
  borderLeftWidth: 'min(1px, var(--arrow-visible-size))',
  borderTopWidth: 'min(1px, var(--arrow-visible-size))',
  clipPath: 'polygon(0 0, calc(100% + 4.5px) 0, 0 calc(100% + 4.5px))',
  pointerEvents: 'auto',
};

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
      // Page tips sit beneath overlays, dialog backdrops (modal - 1) and toasts; Tooltip raises tips inside overlays.
      zIndex: 'calc({zIndex.modal} - 2)',
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
      // Zag sets these a frame after opening; off-screen by default so an early close never fades at the origin.
      '--x': '0px',
      '--y': '-100vh',
      // Zag lets the pointer through the content but not the positioner, which would take a neighbour's click.
      pointerEvents: 'none',
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
      // The arrow stacks below the content's children, within the content.
      isolation: 'isolate',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
      maxWidth: '18rem',
    },
    arrow: {
      ...chakraSlotRecipes.hoverCard.base?.arrow,
      ...arrowClearOfCorners('{sizes.3}'),
      ...arrowInsideContent,
    },
    arrowTip: {
      ...chakraSlotRecipes.hoverCard.base?.arrowTip,
      borderColor: 'border.emphasized',
      ...arrowTipOutsideContent,
    },
  },
  defaultVariants: { size: 'xs' },
});

export const popoverSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.popover,
  base: {
    ...chakraSlotRecipes.popover.base,
    content: {
      ...chakraSlotRecipes.popover.base?.content,
      '--popover-bg': 'colors.bg.muted',
      isolation: 'isolate',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
      _open: { animationStyle: 'slide-fade-in', animationDuration: 'faster' },
      _closed: { animationStyle: 'slide-fade-out', animationDuration: 'faster' },
    },
    arrow: {
      ...chakraSlotRecipes.popover.base?.arrow,
      ...arrowClearOfCorners('{sizes.3}'),
      ...arrowInsideContent,
    },
    arrowTip: {
      ...chakraSlotRecipes.popover.base?.arrowTip,
      borderColor: 'border.emphasized',
      ...arrowTipOutsideContent,
    },
  },
});

/** White status text stays AA: success and warning use 700 fills, since Chakra's green/orange 600 are too light. */
export const toastSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.toast,
  base: {
    ...chakraSlotRecipes.toast.base,
    root: {
      ...chakraSlotRecipes.toast.base?.root,
      // The 700 steps need white text in dark modes, and Chakra's lightening hover fails AA on red, so hover darkens.
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

/** Each step subtracts the root border and inset from the shared control height. */
const SEGMENT_BASIS: Record<ControlSize, string> = { ...CONTROL_BASIS, xl: 'lg', '2xl': '2xl' };

const SEGMENT_SIZE_TWEAKS: Partial<Record<ControlSize, SystemStyleObject>> = {
  xs: { px: '1.5' },
  sm: { px: '2' },
  md: { px: '2.5' },
  lg: { px: '3.5', textStyle: 'md' },
};

// Match SegmentTabs: a neutral selection, the strip's hover fill and its dividers.
const SEGMENT_SELECTED_BG = 'gray.hoverTint/15';
// Raw selectors skip disabled items, as the theme's `_hover` condition does.
const SEGMENT_HOVERED = '&:not([data-disabled]):hover';

export const segmentGroupSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.segmentGroup,
  base: {
    ...chakraSlotRecipes.segmentGroup.base,
    root: {
      ...chakraSlotRecipes.segmentGroup.base?.root,
      // Concentric with the root: the 5px control radius less the 1px border and 2px inset.
      '--segment-radius': 'radii.xs',
      '--segment-inset': 'spacing.0.5',
      '--segment-gap': 'spacing.0.5',
      '--segment-indicator-shadow': 'none',
      bg: 'transparent',
      borderColor: 'border.subtle',
      borderRadius: 'control',
      borderWidth: '1px',
      boxShadow: 'none',
      gap: 'var(--segment-gap)',
      p: 'var(--segment-inset)',
    },
    item: {
      ...chakraSlotRecipes.segmentGroup.base?.item,
      color: 'fg.muted',
      // Use positive z-order; axe can treat negative indicators as beneath the page and measure the wrong contrast.
      zIndex: 1,
      fontWeight: '500',
      transitionDuration: 'faster',
      transitionProperty: 'background, color',
      _before: {
        content: '""',
        position: 'absolute',
        bg: 'border.emphasized',
        borderRadius: 'full',
        transitionDuration: 'faster',
        transitionProperty: 'opacity',
      },
      _horizontal: {
        _before: {
          insetInlineStart: 'calc(var(--segment-gap) / -2 - 0.5px)',
          insetBlock: 'auto',
          top: '50%',
          translate: '0 -50%',
          width: '1px',
          height: '3.5',
        },
      },
      _vertical: {
        _before: {
          insetBlockStart: 'calc(var(--segment-gap) / -2 - 0.5px)',
          insetInline: '1.5',
          height: '1px',
        },
      },
      [`${SEGMENT_HOVERED}, ${SEGMENT_HOVERED} + &`]: { _before: { opacity: '0' } },
      _checked: { color: 'fg' },
      _hover: {
        '&:not([data-state=checked])': { bg: 'bg.hover', color: 'fg' },
      },
      '&[data-state=checked][data-ssr]': {
        bg: SEGMENT_SELECTED_BG,
        shadow: 'none',
      },
    },
    indicator: {
      ...chakraSlotRecipes.segmentGroup.base?.indicator,
      '--transition-duration': '{durations.fast}',
      bg: SEGMENT_SELECTED_BG,
      shadow: 'none',
      zIndex: 0,
    },
  },
  variants: {
    ...chakraSlotRecipes.segmentGroup.variants,
    size: controlSizes<SlotStyles>(
      chakraSlotRecipes.segmentGroup.variants?.size,
      (styles, height, size) => ({
        item: {
          ...styles?.item,
          height: `calc(${height} - 2px - 2 * var(--segment-inset))`,
          ...SEGMENT_SIZE_TWEAKS[size],
        },
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
    // The steppers fill the root inside the field's border. The stock height follows the size's `--input-height`,
    // so steppers would stand taller than a field sized apart from its root's size. The root keeps to its content,
    // so a stretching flex or grid parent cannot make it taller than the field.
    root: {
      ...chakraSlotRecipes.numberInput.base?.root,
      height: 'fit-content',
    },
    control: {
      ...chakraSlotRecipes.numberInput.base?.control,
      height: 'calc(100% - 2px)',
    },
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

/** A context menu anchors to a point, so its arrow has no visible size until Zag reports the anchor's size. */
const MENU_ARROW_SIZE =
  'min({sizes.3}, max(0px, calc((min(var(--reference-width, 2px), var(--reference-height, 2px)) - 1px) * 1000)))';

export const menuSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.menu,
  base: {
    ...chakraSlotRecipes.menu.base,
    content: {
      ...chakraSlotRecipes.menu.base?.content,
      ...dropdownContent,
      // Names the surface for content drawn against it, as `--popover-bg` does for popovers.
      '--menu-bg': `colors.${dropdownContent.bg}`,
    },
    arrow: {
      ...chakraSlotRecipes.menu.base?.arrow,
      ...arrowClearOfCorners(MENU_ARROW_SIZE),
      // Beside the content, the arrow is outside the scope of its `--menu-bg`.
      '--arrow-background': `colors.${dropdownContent.bg}`,
      // MenuContent mirrors the content's state and placement onto the arrow, so it slides and fades with the content.
      _open: chakraSlotRecipes.menu.base?.content?._open,
      // Held at its last frame: it ends with the content's exit, which unmounts both.
      _closed: { ...chakraSlotRecipes.menu.base?.content?._closed, animationFillMode: 'forwards' },
      '[hidden] + &': { display: 'none' },
    },
    arrowTip: {
      ...chakraSlotRecipes.menu.base?.arrowTip,
      borderColor: dropdownContent.borderColor,
      ...arrowTipOutsideContent,
    },
    item: {
      ...chakraSlotRecipes.menu.base?.item,
      ...dropdownItem,
      // Item icons sit back from the label; danger items and deeper glyphs keep their own color.
      '&:not([data-danger]) > svg': { color: 'fg.subtle' },
    },
    itemGroupLabel: {
      ...chakraSlotRecipes.menu.base?.itemGroupLabel,
      ...dropdownGroupLabel,
    },
    separator: {
      ...chakraSlotRecipes.menu.base?.separator,
      // The rule's top border is the line; a fill under it rounds to a second device pixel at fractional display scales.
      bg: 'transparent',
      borderColor: 'border.subtle',
    },
  },
});

/** Stars are the hit targets; the stock recipe gives them no focus ring or pointer. */
export const ratingGroupSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.ratingGroup,
  base: {
    ...chakraSlotRecipes.ratingGroup.base,
    item: {
      ...chakraSlotRecipes.ratingGroup.base?.item,
      borderRadius: 'l1',
      cursor: 'pointer',
      outline: 0,
      _focusVisible: { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' },
      _disabled: { cursor: 'not-allowed', layerStyle: 'disabled' },
    },
  },
});

export const selectSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.select,
  // Override outline _expanded at variant level, where base styles lose to it.
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

/** Dialogs stack below menus: zag copies a menu's z-index to its positioner before the layer index lands. */
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

export const scrollAreaSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.scrollArea,
  base: {
    ...chakraSlotRecipes.scrollArea.base,
    scrollbar: {
      ...chakraSlotRecipes.scrollArea.base?.scrollbar,
      // Above sticky content (z-index 1) so opaque pinned headers never cover the thumb.
      zIndex: 2,
    },
  },
});

/** A 6px invisible outset gives the 12px control a 24px target without reaching a neighbour 8px away. */
export const checkboxSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.checkbox,
  base: {
    ...chakraSlotRecipes.checkbox.base,
    control: {
      ...chakraSlotRecipes.checkbox.base?.control,
      position: 'relative',
      transitionDuration: 'fast',
      transitionProperty: 'border-color, background',
      _before: { content: '""', inset: 'calc(-1px - {spacing.1.5})', position: 'absolute' },
    },
  },
  variants: {
    ...chakraSlotRecipes.checkbox.variants,
    variant: {
      ...chakraSlotRecipes.checkbox.variants?.variant,
      solid: {
        control: {
          ...chakraSlotRecipes.checkbox.variants?.variant?.solid?.control,
          _hover: {
            // Halfway to the foreground beats the rest border in every theme, high contrast included.
            '&[data-state=unchecked]:not([data-invalid])': {
              borderColor: 'color-mix(in oklab, {colors.border.emphasized}, {colors.fg})',
            },
            '&:is([data-state=checked], [data-state=indeterminate]):not([data-invalid])': {
              bg: 'colorPalette.solid/90',
              borderColor: 'colorPalette.solid/90',
            },
          },
        },
      },
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
      // Zag already applies half-thumb offsets, so the marker inset is zero.
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

/** The emphasized border stays readable on row hover fills; each size is 2px shorter than stock to fit a text line. */
const KBD_HEIGHTS: Record<string, string> = { '4.5': '16px', '5': '18px', '6': '22px' };

export const kbdRecipe = defineRecipe({
  ...chakraRecipes.kbd,
  variants: {
    ...chakraRecipes.kbd.variants,
    variant: {
      ...chakraRecipes.kbd.variants?.variant,
      raised: { ...chakraRecipes.kbd.variants?.variant?.raised, borderColor: 'border.emphasized' },
      outline: { ...chakraRecipes.kbd.variants?.variant?.outline, borderColor: 'border.emphasized' },
    },
    // Keyed by stock height, as the size names are already rebased onto the workbench scale.
    size: Object.fromEntries(
      Object.entries(chakraRecipes.kbd.variants?.size ?? {}).map(([name, styles]) => [
        name,
        { ...styles, height: KBD_HEIGHTS[String((styles as { height?: unknown }).height)] ?? styles.height },
      ])
    ),
  } as unknown as typeof chakraRecipes.kbd.variants,
});

/** A linear sweep avoids endpoint pauses; system.ts disables skeleton motion and gradients under reduced motion. */
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

/** Inset so neighbouring rows never clip it; exported for row-like surfaces such as tree items. */
export const rowFocusRing = {
  outline: '2px solid',
  outlineColor: 'accent.solid',
  outlineOffset: '-2px',
} as const;

/** Hover, extended to a row whose context menu is open so the row it acts on stays marked. */
const ROW_POINTED = '&:is(:hover, [data-hover], [data-menu-open]):not(:disabled, [data-disabled], [data-static])';

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

/** The checkbox sits beside the primary button, never inside it; accent recolors muted text for the solid fill. */
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
      flexWrap: 'wrap',
      minW: 0,
      position: 'relative',
      '&:has([data-list-primary]:focus-visible)': rowFocusRing,
      // A disabled row stays focusable, so its content dims instead of the root: the focus ring keeps full strength.
      _disabled: { cursor: 'not-allowed', '& > *': { opacity: 0.5 } },
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
    actions: {
      alignItems: 'center',
      display: 'flex',
      flexShrink: 0,
      gap: '0.5',
      pe: '1',
    },
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
      snug: {
        root: { minH: '12' },
        primary: { gap: '2.5', px: '1.5', py: '1.5' },
        detail: { pb: '1.5', px: '1.5' },
      },
    },
  },
  defaultVariants: { active: 'none', density: 'regular' },
});

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
  variants: {
    placement: {
      list: {},
      bar: { root: { alignItems: 'center', pb: '0' } },
    },
  },
  defaultVariants: { placement: 'list' },
});

/** Trimmed to cap height so flex centring aligns letters with the icon; without `text-box` the full line box stays. */
export const chipLabelStyles = {
  '@supports (text-box: trim-both cap alphabetic)': {
    paddingBlock: '0.3em',
    textBox: 'trim-both cap alphabetic',
  },
} as const;

export const chipRecipe = defineRecipe({
  base: {
    alignItems: 'center',
    // Fill the host row, so a progress hairline keeps to its bottom edge however tall the trimmed label is.
    alignSelf: 'stretch',
    // The status bar's hover and active fill radius, which the progress hairline is clipped to.
    borderRadius: 'control',
    display: 'inline-flex',
    flexShrink: '0',
    fontSize: 'xs',
    fontWeight: '500',
    gap: '1.5',
    px: '2',
    py: '0.5',
    whiteSpace: 'nowrap',
    '& > p': chipLabelStyles,
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

export const dataListSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.dataList,
  base: {
    ...chakraSlotRecipes.dataList.base,
    item: {
      ...chakraSlotRecipes.dataList.base?.item,
      '&:not(:first-of-type)': {
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
  '[data-line-hidden] > &::after': { borderColor: 'accent.solid' },
} as const;

/** The hit strip extends toward the end side, away from the previous pane's scrollbar. */
export const resizeHandleSlotRecipe = defineSlotRecipe({
  slots: ['root', 'handle'],
  base: {
    root: {
      flexShrink: '0',
      position: 'relative',
      // Above region outlines (zIndex 4) so the capsule masks an outline passing through it.
      zIndex: 5,
      // A border, not a 1px background: backgrounds can round to two device pixels at fractional scales.
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
