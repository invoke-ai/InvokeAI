import { createSystem, defineConfig } from '@chakra-ui/react';

import { baseConfig } from './rebase';
import {
  buttonRecipe,
  checkboxSlotRecipe,
  colorPickerSlotRecipe,
  comboboxSlotRecipe,
  dataListSlotRecipe,
  dialogSlotRecipe,
  hoverCardSlotRecipe,
  inputRecipe,
  kbdRecipe,
  menuSlotRecipe,
  numberInputSlotRecipe,
  popoverSlotRecipe,
  progressCircleSlotRecipe,
  ratingGroupSlotRecipe,
  scrollAreaSlotRecipe,
  segmentGroupSlotRecipe,
  selectSlotRecipe,
  skeletonRecipe,
  sliderSlotRecipe,
  tabsSlotRecipe,
  toastSlotRecipe,
  textareaRecipe,
  tooltipSlotRecipe,
} from './recipes';
import { CONTROL_HEIGHT_PX } from './scale';
import { DEFAULT_THEME, DEFAULT_THEME_ID, type NeutralStep, THEMES, type ThemeDefinition } from './themes';

/** Theme-varying ramps emit as semantic tokens; root data-theme switches appearance. */

const NON_DEFAULT_THEMES = THEMES.filter((theme) => theme.id !== DEFAULT_THEME_ID);

const conditionName = (id: string): string => `theme${id.charAt(0).toUpperCase()}${id.slice(1)}`;

type TokenValue = { value: Record<string, string> };
type Compute = (theme: ThemeDefinition) => string;

const colorToken = (compute: Compute): TokenValue => {
  const value: Record<string, string> = { base: compute(DEFAULT_THEME) };
  for (const theme of NON_DEFAULT_THEMES) {
    value[`_${conditionName(theme.id)}`] = compute(theme);
  }
  return { value };
};

const mix = (top: Compute, pct: number, bottom: Compute): TokenValue =>
  colorToken((theme) => `color-mix(in oklab, ${top(theme)} ${pct}%, ${bottom(theme)})`);

const LIGHT_FALLBACK_THEME = THEMES.find((theme) => theme.colorScheme === 'light') ?? DEFAULT_THEME;

/** Shadows Chakra's default gray _light/_dark keys, which nested palettes deep-merge. */
const grayToken = (compute: Compute): TokenValue => {
  const token = colorToken(compute);
  token.value._light = compute(LIGHT_FALLBACK_THEME);
  token.value._dark = compute(DEFAULT_THEME);
  return token;
};

/** Use explicit data-theme conditions: Chakra _light includes :root and can leak light values into dark themes. */
const ref = (step: NeutralStep): string => `{colors.neutral.${step}}`;
const stepRef = (darkStep: NeutralStep, lightStep: NeutralStep): TokenValue =>
  colorToken((theme) => ref(theme.colorScheme === 'light' ? lightStep : darkStep));

/** App high-contrast conditions follow color scheme; they are separate from forced-colors. */
const withHighContrast = (token: TokenValue, darkStep: NeutralStep, lightStep: NeutralStep): TokenValue => {
  token.value._highContrastDark = ref(darkStep);
  token.value._highContrastLight = ref(lightStep);
  return token;
};

const contrastStepRef = (
  darkStep: NeutralStep,
  lightStep: NeutralStep,
  highDarkStep: NeutralStep,
  highLightStep: NeutralStep
): TokenValue => withHighContrast(stepRef(darkStep, lightStep), highDarkStep, highLightStep);

const STEPS: NeutralStep[] = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

const surface: Compute = (theme) =>
  theme.colorScheme === 'light' ? theme.colors.neutral[50] : theme.colors.neutral[900];

const danger: Compute = (theme) => theme.colors.danger;
const success: Compute = (theme) => theme.colors.success;
const warning: Compute = (theme) => theme.colors.warning;
const brandSolid: Compute = (theme) => theme.colors.brand.solid;

/** Darken brand.fg on light themes for contrast; keep bright brand.solid paired with brand.contrast. */
const brandFg: Compute = (theme) =>
  theme.colorScheme === 'light'
    ? `color-mix(in oklab, ${theme.colors.brand.solid} 50%, ${theme.colors.neutral[950]})`
    : theme.colors.brand.solid;
const accentSolid: Compute = (theme) => theme.colors.accent.solid;
const hoverTint: Compute = (theme) =>
  `color-mix(in oklab, ${theme.colors.accent.solid} ${theme.interactionTint}%, ${
    theme.colorScheme === 'light' ? theme.colors.neutral[950] : theme.colors.neutral[50]
  })`;

const neutralRamp = Object.fromEntries(STEPS.map((step) => [step, colorToken((theme) => theme.colors.neutral[step])]));

const categoricalPalettes = Object.fromEntries(
  Object.entries({
    aqua: '#5cc8d8',
    azure: '#3fa9e8',
    coral: '#ff8e80',
    gold: '#f6d365',
    ice: '#b3e5fc',
    lavender: '#bd9fff',
    lime: '#c5e17a',
    mint: '#7fd8b5',
    periwinkle: '#a5b4fc',
    sage: '#a3b18a',
    silver: '#d4d4d5',
    steel: '#94aec9',
  }).map(([name, seed]) => {
    const fg: Compute = (theme) =>
      theme.colorScheme === 'light' ? `color-mix(in oklab, ${seed} 40%, ${theme.colors.neutral[950]})` : seed;
    return [
      name,
      {
        // The hue itself in every theme, for identity marks (color labels, rating stars); edge it with `fg` where it
        // must stand out from a light surface.
        swatch: { value: seed },
        fg: colorToken(fg),
        solid: colorToken(fg),
        contrast: stepRef(950, 50),
        subtle: mix(fg, 12, surface),
        muted: mix(fg, 24, surface),
        emphasized: mix(fg, 36, surface),
        border: colorToken(fg),
        focusRing: colorToken(fg),
        hoverTint: colorToken(fg),
      },
    ];
  })
);

/** Reuse Chakra token names so built-ins inherit themes. */
const semanticColors = {
  ...categoricalPalettes,
  neutral: neutralRamp,

  // Light panels are brighter than the app background, not a reversed dark ladder.
  bg: stepRef(950, 200),
  'bg.subtle': stepRef(900, 50),
  'bg.muted': stepRef(800, 100),
  'bg.panel': stepRef(800, 100),
  'bg.emphasized': colorToken((theme) => theme.colors.control),
  'bg.inset': colorToken((theme) => theme.colors.inset),
  /** Matches the ghost/outline button hover (`gray.hoverTint/10`), so an open trigger and its option match. */
  'bg.hover': mix(hoverTint, 10, () => 'transparent'),
  'bg.error': mix(danger, 14, surface),
  'bg.success': mix(success, 14, surface),
  'bg.warning': mix(warning, 14, surface),

  fg: stepRef(50, 950),
  'fg.muted': contrastStepRef(300, 700, 200, 800),
  'fg.subtle': contrastStepRef(400, 500, 300, 700),
  'fg.grid': colorToken((theme) => theme.colors.grid),
  'fg.error': colorToken(danger),
  'fg.success': colorToken(success),
  'fg.warning': colorToken(warning),

  border: contrastStepRef(600, 300, 300, 500),
  'border.subtle': contrastStepRef(600, 300, 300, 500),
  'border.muted': contrastStepRef(600, 300, 300, 500),
  'border.emphasized': contrastStepRef(500, 400, 200, 600),
  'border.error': colorToken(danger),
  'border.image': colorToken((theme) => (theme.colorScheme === 'light' ? 'oklch(0 0 0 / 0.1)' : 'oklch(1 0 0 / 0.1)')),

  /** Nested for Chakra's virtual-token map. */
  gray: {
    contrast: grayToken((theme) =>
      theme.colorScheme === 'light' ? theme.colors.neutral[200] : theme.colors.neutral[950]
    ),
    fg: grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[950] : theme.colors.neutral[50])),
    subtle: grayToken((theme) => theme.colors.fill),
    muted: grayToken((theme) => theme.colors.control),
    emphasized: withHighContrast(
      grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[400] : theme.colors.neutral[500])),
      300,
      600
    ),
    solid: grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[950] : theme.colors.neutral[50])),
    focusRing: grayToken(accentSolid),
    border: withHighContrast(
      grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[400] : theme.colors.neutral[500])),
      300,
      600
    ),
    hoverTint: grayToken(hoverTint),
  },
  red: { hoverTint: { value: '{colors.red.fg}' } },
  orange: { hoverTint: { value: '{colors.orange.fg}' } },
  green: { hoverTint: { value: '{colors.green.fg}' } },
  blue: { hoverTint: { value: '{colors.blue.fg}' } },
  /** Use brand.fg for text/icons and brand.solid with brand.contrast for fills. */
  brand: {
    solid: colorToken(brandSolid),
    contrast: colorToken((theme) => theme.colors.brand.contrast),
    fg: colorToken(brandFg),
    subtle: mix(brandSolid, 16, surface),
    muted: mix(brandSolid, 26, surface),
    emphasized: mix(brandSolid, 36, surface),
    focusRing: colorToken(accentSolid),
    border: colorToken(brandSolid),
    hoverTint: colorToken(brandFg),
  },
  accent: {
    solid: colorToken(accentSolid),
    contrast: colorToken((theme) => theme.colors.accent.contrast),
    fg: colorToken(accentSolid),
    subtle: mix(accentSolid, 16, surface),
    muted: mix(accentSolid, 26, surface),
    emphasized: mix(accentSolid, 36, surface),
    focusRing: colorToken(accentSolid),
    border: colorToken(accentSolid),
    hoverTint: colorToken(accentSolid),
  },
};

// :root specificity makes explicit themes override .dark/.light fallbacks.
const themeConditions = Object.fromEntries(
  NON_DEFAULT_THEMES.map((theme) => [conditionName(theme.id), `:root[data-theme=${theme.id}]`])
);
// Paired attributes outrank data-theme; enumerate light themes to avoid dark fallbacks.
const lightThemeSelectors = THEMES.filter((theme) => theme.colorScheme === 'light')
  .map((theme) => `[data-theme=${theme.id}]`)
  .join(', ');
const highContrastConditions = {
  highContrastDark: `:root[data-high-contrast=true]:not(${lightThemeSelectors})`,
  highContrastLight: `:root[data-high-contrast=true]:is(${lightThemeSelectors})`,
};

/** 0 before the collapsible measures (0px), so the scale is 1, never infinite or negative. */
const insetRatio = (inset: string, size: string): string =>
  `tan(atan2(min(${inset}, ${size} * 1000), max(${size}, ${inset})))`;
const SECTION_SCALE_INSET = `scale(calc(1 - ${insetRatio('8px', 'var(--width)')}), calc(1 - ${insetRatio('4px', 'var(--height)')}))`;

const motionDurationToken = (base: string): TokenValue => ({ value: { base, _reduceMotion: '1ms' } });
const motionAnimationToken = (base: string): TokenValue => ({ value: { base, _reduceMotion: 'none' } });

/** Chakra excludes only native and `data-disabled` controls; `aria-disabled` ones must not respond or look expanded. */
const interactionConditions = {
  hover: ['@media (hover: hover)', '&:is(:hover, [data-hover]):not(:disabled, [data-disabled], [aria-disabled=true])'],
  active: '&:is(:active, [data-active]):not(:disabled, [data-disabled], [aria-disabled=true], [data-state=open])',
  expanded: '&:is([aria-expanded=true], [data-expanded], [data-state=expanded]):not([aria-disabled=true])',
};

const config = defineConfig({
  conditions: {
    ...themeConditions,
    ...highContrastConditions,
    ...interactionConditions,
    reduceMotion: ':root[data-reduce-motion=true]',
  },
  globalCss: {
    'html, body, #root': {
      height: '100%',
    },
    body: {
      bg: 'bg',
      color: 'fg',
      fontFamily: 'body',
      margin: 0,
      overflow: 'hidden',
      // Only the size: the md text style's absolute line height would be inherited by larger text.
      fontSize: 'md',
    },
    // Keep role values unquoted while SamOptions assertions match quoted markup.
    'button, [role=button], [role=menuitem], [role=menuitemcheckbox], [role=menuitemradio], [role=option], [role=tab], [role=radio], [role=checkbox], [role=switch]':
      {
        cursor: 'default',
      },
    // Apply the drag cursor to descendants too, overriding controls' own cursors during gallery drags.
    'body[data-gallery-drag], body[data-gallery-drag] *': {
      cursor: 'grabbing !important',
    },
    // A pointer drag (resize, window move) owns the cursor and selection until it ends; iframes would swallow moves.
    ':root[data-pointer-drag], :root[data-pointer-drag] *': {
      cursor: 'var(--wb-pointer-drag-cursor) !important',
      userSelect: 'none !important',
    },
    ':root[data-pointer-drag] iframe': {
      pointerEvents: 'none',
    },
    ':root': {
      '--wb-motion-duration-fast': '0.12s',
      '--wb-motion-duration-medium': '0.15s',
      '--wb-motion-duration-slow': '0.2s',
      '--wb-motion-animation-iteration-count': 'infinite',
    },
    // Keep reduced-motion durations nonzero because Ark presence awaits animation events.
    ':root[data-reduce-motion="true"]': {
      '--wb-motion-duration-fast': '1ms',
      '--wb-motion-duration-medium': '1ms',
      '--wb-motion-duration-slow': '1ms',
      '--wb-motion-animation-iteration-count': '1',
      scrollBehavior: 'auto !important',
    },
    ':root[data-reduce-motion="true"] .chakra-skeleton': {
      animation: 'none !important',
      backgroundImage: 'none !important',
    },
    ':root[data-reduce-motion="true"] .chakra-spinner': {
      animation: 'spin 2s linear infinite !important',
    },
  },
  theme: {
    keyframes: {
      'wb-status-sweep': {
        from: { transform: 'translateX(-100%)' },
        to: { transform: 'translateX(300%)' },
      },
      // Runs on the fields, never the content: Chromium leaves fields unpainted when one element animates a composited
      // transform and its height under `overflow: hidden`.
      'wb-section-scale-in': {
        from: { transform: SECTION_SCALE_INSET },
        to: { transform: 'scale(1)' },
      },
      'wb-section-scale-out': {
        from: { transform: 'scale(1)' },
        to: { transform: SECTION_SCALE_INSET },
      },
    },
    layerStyles: {
      // Matches the workflow editor's grid at 100% zoom (1.5px dots every 25px).
      dotGrid: {
        value: {
          backgroundColor: 'bg.inset',
          backgroundImage: 'radial-gradient(circle, currentColor 0.4px, transparent 1px)',
          backgroundPosition: 'center',
          backgroundRepeat: 'repeat',
          backgroundSize: '25px 25px',
          color: 'fg.grid',
        },
      },
    },
    tokens: {
      cursor: {
        button: { value: 'default' },
        switch: { value: 'default' },
      },
      radii: {
        control: { value: '0.3125rem' },
      },
      sizes: {
        control: Object.fromEntries(
          Object.entries(CONTROL_HEIGHT_PX).map(([size, px]) => [size, { value: `${px / 16}rem` }])
        ),
      },
      fonts: {
        body: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
        heading: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
        },
      },
    },
    semanticTokens: {
      animations: {
        bounce: motionAnimationToken('bounce 1s infinite'),
        ping: motionAnimationToken('ping 1s cubic-bezier(0, 0, 0.2, 1) infinite'),
        pulse: motionAnimationToken('pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'),
        spin: motionAnimationToken('spin 1s linear infinite'),
      },
      colors: semanticColors,
      durations: {
        fastest: motionDurationToken('50ms'),
        faster: motionDurationToken('100ms'),
        fast: motionDurationToken('150ms'),
        moderate: motionDurationToken('200ms'),
        slow: motionDurationToken('300ms'),
        slower: motionDurationToken('400ms'),
        slowest: motionDurationToken('500ms'),
      },
    },
    recipes: {
      button: buttonRecipe,
      input: inputRecipe,
      kbd: kbdRecipe,
      skeleton: skeletonRecipe,
      textarea: textareaRecipe,
    },
    slotRecipes: {
      checkbox: checkboxSlotRecipe,
      colorPicker: colorPickerSlotRecipe,
      combobox: comboboxSlotRecipe,
      dataList: dataListSlotRecipe,
      dialog: dialogSlotRecipe,
      hoverCard: hoverCardSlotRecipe,
      menu: menuSlotRecipe,
      numberInput: numberInputSlotRecipe,
      popover: popoverSlotRecipe,
      progressCircle: progressCircleSlotRecipe,
      ratingGroup: ratingGroupSlotRecipe,
      scrollArea: scrollAreaSlotRecipe,
      segmentGroup: segmentGroupSlotRecipe,
      select: selectSlotRecipe,
      slider: sliderSlotRecipe,
      tabs: tabsSlotRecipe,
      toast: toastSlotRecipe,
      tooltip: tooltipSlotRecipe,
    },
  },
});

export const system = createSystem(baseConfig, config);

export { THEMES, THEMES_BY_ID, DEFAULT_THEME, DEFAULT_THEME_ID, previewSwatches } from './themes';
export type { ThemeColors, ThemeDefinition, NeutralStep } from './themes';
