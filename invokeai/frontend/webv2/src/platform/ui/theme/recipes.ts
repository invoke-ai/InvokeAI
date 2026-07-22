import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';
import { recipes as chakraRecipes, slotRecipes as chakraSlotRecipes } from '@chakra-ui/react/theme';

/**
 * Reusable, theme-aware recipes for the workbench design system.
 *
 * Recipes reference semantic tokens only, so every variant automatically tracks
 * the active theme. Two kinds live here:
 *
 *   1. Overrides for Chakra's built-in component recipes (`tooltip`, `menu`,
 *      `select`, `combobox`, `dialog`) — registered in `system.ts` so every
 *      instance app-wide gets the workbench chrome with zero props at the call
 *      site.
 *   2. Workbench-specific recipes (`panel`, `row`, `chip`, `fieldLabel`,
 *      `themeCard`) — consumed via the wrappers in `platform/ui`
 *      with `useRecipe({ recipe })` / `useSlotRecipe({ recipe })`, which keeps
 *      them fully typed without the Chakra typegen step.
 *
 * Either way, this file is the single place where shared component styling is
 * edited.
 */

/* -------------------------------------------------------------------------- *
 * Built-in component overrides (registered in system.ts)
 * -------------------------------------------------------------------------- */

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
 * Tabs: quick neutral hover feedback, with accent reserved for selection.
 * Restricting hover styles to unselected triggers keeps active tabs visually
 * stable without relying on condition ordering in the generated CSS.
 */
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
  } as typeof chakraSlotRecipes.tabs.variants,
});

/**
 * Shared interactive states for every bordered form control — text inputs,
 * textareas, number inputs, select/combobox triggers, and hand-rolled trigger
 * buttons (ModelSelect). Idle border is `border`; hover steps to
 * `border.emphasized`; an open dropdown commits to `accent.solid` so hover and
 * open remain distinguishable. Focus uses the accent border without adding a
 * second ring around the control.
 */
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

/** Text input: Chakra default plus the shared hover/transition treatment. */
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

/** Textarea: Chakra default plus the shared hover/transition treatment. */
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

/** Number input: the input slot gets the shared hover/transition treatment. */
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

/** Shared dropdown surface: one look for menu, select, combobox, and hand-rolled poppers (ModelSelect). */
export const dropdownContent = {
  bg: 'bg.muted',
  borderColor: 'border.emphasized',
  borderRadius: 'md',
  borderWidth: '1px',
  boxShadow: 'lg',
  color: 'fg',
};

/**
 * Shared dropdown item treatment. The highlight sits on `bg.emphasized`: the
 * content surface is `bg.muted`, and on the dark ramps `bg.subtle` is darker
 * than the surface, which made the old highlight nearly invisible (classic
 * theme especially). `bg.emphasized` is one step away from the surface in
 * every theme, so the highlight reads everywhere.
 */
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

/** Menu / context-menu chrome: popover surface with an emphasized stroke. */
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

/** Select dropdown chrome: same surface and item treatment as menus. */
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

/** Combobox chrome: kept aligned with Select for future searchable fields. */
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

/** Dialog chrome: panel surface with a hairline stroke. */
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

/** Slider marks: compact auxiliary labels for dense widget controls. */
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

/** Progress circle: add a compact icon-sized variant for dense chrome like tabs. */
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

/* -------------------------------------------------------------------------- *
 * Reusable UI recipes (consumed through Platform UI wrappers)
 * -------------------------------------------------------------------------- */

/** Bordered surface container — panels, cards, wells. */
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

/** Interactive list / table row with hover, focus, and active fills. */
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
    _hover: { bg: 'bg.muted' },
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

/** Compact status chip with intent tones — queue counts, server state, versions. */
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

/** Compact uppercase field label shared across widget forms and the settings modal. */
export const fieldLabelRecipe = defineRecipe({
  base: {
    color: 'fg.muted',
    fontSize: '2xs',
    fontWeight: '600',
    letterSpacing: '0.03em',
  },
});

/** Selectable theme swatch card used by the Settings appearance picker. */
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
