import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';
import { slotRecipes as chakraSlotRecipes } from '@chakra-ui/react/theme';

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
 *      `themeCard`) — consumed via the wrappers in `workbench/components/ui`
 *      with `useRecipe({ recipe })` / `useSlotRecipe({ recipe })`, which keeps
 *      them fully typed without the Chakra typegen step.
 *
 * Either way, this file is the single place where shared component styling is
 * edited.
 */

/* -------------------------------------------------------------------------- *
 * Built-in component overrides (registered in system.ts)
 * -------------------------------------------------------------------------- */

/** Tooltip chrome: raised surface with a hairline stroke instead of inverted fill. */
export const tooltipSlotRecipe = defineSlotRecipe({
  slots: ['content', 'arrowTip'],
  base: {
    content: {
      '--tooltip-bg': 'colors.bg.muted',
      bg: 'var(--tooltip-bg)',
      borderColor: 'border.emphasized',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg',
    },
    arrowTip: {
      borderColor: 'border.emphasized',
    },
  },
});

const dropdownContent = {
  bg: 'bg.muted',
  borderColor: 'border.emphasized',
  borderRadius: 'lg',
  borderWidth: '1px',
  boxShadow: 'lg',
  color: 'fg',
};

const dropdownItem = {
  _highlighted: { bg: 'bg.subtle' },
  _focusVisible: { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' },
};

const dropdownGroupLabel = {
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
});

/** Select dropdown chrome: same surface and item treatment as menus. */
export const selectSlotRecipe = defineSlotRecipe({
  ...chakraSlotRecipes.select,
  base: {
    ...chakraSlotRecipes.select.base,
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
  base: {
    ...chakraSlotRecipes.combobox.base,
    content: {
      ...chakraSlotRecipes.combobox.base?.content,
      ...dropdownContent,
    },
    input: {
      ...chakraSlotRecipes.combobox.base?.input,
      _hover: { borderColor: 'border.emphasized' },
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
 * Workbench recipes (consumed through workbench/components/ui wrappers)
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
    _focusVisible: { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' },
    _disabled: { cursor: 'not-allowed', opacity: 0.5 },
  },
  variants: {
    active: {
      none: {},
      muted: { bg: 'bg.muted' },
      brand: { bg: 'brand.subtle', color: 'brand.fg', _hover: { bg: 'brand.subtle' } },
      accent: { bg: 'accent.solid', color: 'accent.contrast', _hover: { bg: 'accent.solid' } },
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
    letterSpacing: '0.02em',
    textTransform: 'uppercase',
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
      _focusVisible: { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '2px' },
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
    body: { alignItems: 'flex-start', display: 'flex', flexDirection: 'column', gap: '0.5' },
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
