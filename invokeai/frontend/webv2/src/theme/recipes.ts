import { defineRecipe, defineSlotRecipe } from '@chakra-ui/react';

/**
 * Reusable, theme-aware recipes for the workbench design system.
 *
 * Recipes reference semantic tokens only, so every variant automatically tracks
 * the active theme. They are consumed inline with `useRecipe({ recipe })` /
 * `useSlotRecipe({ recipe })`, which keeps them fully typed without the Chakra
 * typegen step and lets components share one styling contract instead of
 * repeating prop soup.
 */

/** Selectable theme swatch card used by the Settings appearance picker. */
export const themeCardRecipe = defineSlotRecipe({
  slots: ['root', 'preview', 'swatch', 'body', 'name', 'description', 'indicator'],
  base: {
    root: {
      alignItems: 'stretch',
      bg: 'bg.surface',
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
      transition: 'border-color 0.12s ease, background 0.12s ease, transform 0.12s ease',
      _hover: { borderColor: 'border.emphasis' },
      _focusVisible: { outline: '2px solid', outlineColor: 'accent.active', outlineOffset: '2px' },
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
    name: { color: 'fg.default', fontSize: 'sm', fontWeight: '600' },
    description: { color: 'fg.subtle', fontSize: '2xs', lineHeight: '1.3' },
    indicator: {
      alignItems: 'center',
      borderRadius: 'full',
      color: 'accent.invoke',
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
        root: { borderColor: 'accent.invoke', bg: 'bg.surfaceRaised' },
        indicator: { opacity: 1 },
      },
      false: {},
    },
  },
  defaultVariants: { selected: false },
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

/** Shared tooltip chrome for workbench controls. */
export const workbenchTooltipRecipe = defineSlotRecipe({
  slots: ['content', 'arrow', 'arrowTip'],
  base: {
    content: {
      bg: 'bg.surfaceRaised',
      borderColor: 'border.emphasis',
      borderWidth: '1px',
      boxShadow: 'lg',
      color: 'fg.default',
    },
    arrow: {
      '--arrow-background': 'colors.bg.surfaceRaised',
    },
    arrowTip: {
      borderColor: 'border.emphasis',
    },
  },
});
