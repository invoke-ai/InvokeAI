import type { RecipeDefinition, SlotRecipeDefinition, SystemConfig } from '@chakra-ui/react';

import { defaultConfig } from '@chakra-ui/react';
import { recipes as stockRecipes, slotRecipes as stockSlotRecipes } from '@chakra-ui/react/theme';

import { TYPE_SCALE } from './scale';

/**
 * Chakra's stock theme, renamed onto the workbench scales (`./scale`). Chakra sizes for marketing pages, so its stock
 * `md` is 36-40px with 16px text; renaming its variants and font references lets `md` mean the workbench's working
 * size while every stock style keeps its pixels. Overrides in `./recipes` build on these renamed recipes and pin the
 * control heights onto the scale.
 *
 * Chakra's `InputElement` (InputGroup start/end slots) hardcodes the stock `sm` font size outside any recipe, so it
 * now computes 11px; give slot content its own size rather than relying on that inherited one.
 */

type NameMap = Readonly<Record<string, string>>;

/** Stock text style and font size names to the workbench type scale's same-pixel names. */
export const TYPE_NAMES: NameMap = {
  '2xs': 'xs',
  xs: 'md',
  sm: 'lg',
  md: 'xl',
  lg: '2xl',
  xl: '3xl',
  '2xl': '4xl',
  '3xl': '5xl',
  '4xl': '6xl',
  '5xl': '7xl',
  '6xl': '8xl',
  '7xl': '9xl',
};

const SCALE_NAMES = new Set(['3xs', '2xs', 'xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl', '5xl', '6xl', '7xl']);

/** One step: recipes whose stock `sm` is the workbench's usual size. */
const ONE_STEP: NameMap = { xs: 'sm', sm: 'md', md: 'lg', lg: 'xl', xl: '2xl' };

/** Two steps: inline companions whose stock `xs` is the workbench's usual size. */
const TWO_STEPS: NameMap = {
  '3xs': 'xs',
  '2xs': 'sm',
  xs: 'md',
  sm: 'lg',
  md: 'xl',
  lg: '2xl',
  xl: '3xl',
  '2xl': '4xl',
};

/**
 * Stock size variant names to workbench names, per recipe. Control-height recipes map by pixel height onto
 * `CONTROL_HEIGHT_PX` (after `./recipes` pins the heights it overrides); stock steps off that scale are dropped.
 * Scale names a map omits are dropped; other keys (`inherit`, `full`) pass through. Unlisted recipes keep their
 * names because their sizes are widths or paddings (dialog, popover) or the workbench does not use them.
 */
export const SIZE_NAMES: Readonly<Record<string, NameMap>> = {
  alert: ONE_STEP,
  avatar: { '2xs': 'sm', xs: 'lg', sm: 'xl', md: '2xl', lg: '3xl' },
  badge: TWO_STEPS,
  button: { '2xs': 'sm', xs: 'md', sm: 'lg', md: 'xl', lg: '3xl' },
  checkbox: ONE_STEP,
  checkmark: ONE_STEP,
  code: ONE_STEP,
  colorPicker: { '2xs': 'md', xs: 'lg', sm: 'xl', md: '2xl', lg: '3xl' },
  combobox: { xs: 'md', sm: 'lg', md: 'xl' },
  dataList: ONE_STEP,
  heading: TYPE_NAMES,
  icon: TWO_STEPS,
  // Stock 2xs drops: once `./recipes` pins xs to the md height, the two are identical.
  input: { xs: 'md', sm: 'lg', md: 'xl', lg: '3xl' },
  inputAddon: { '2xs': 'md', xs: 'lg', sm: 'xl', md: '2xl', lg: '3xl' },
  kbd: ONE_STEP,
  menu: ONE_STEP,
  nativeSelect: { xs: 'lg', sm: 'xl', md: '2xl', lg: '3xl' },
  numberInput: { xs: 'md', sm: 'lg', md: 'xl', lg: '3xl' },
  progress: TWO_STEPS,
  progressCircle: TWO_STEPS,
  radioGroup: ONE_STEP,
  radiomark: ONE_STEP,
  scrollArea: TWO_STEPS,
  segmentGroup: { xs: 'md', sm: 'lg', md: '2xl', lg: '3xl' },
  select: { xs: 'md', sm: 'lg', md: 'xl' },
  slider: ONE_STEP,
  spinner: TWO_STEPS,
  stat: ONE_STEP,
  status: ONE_STEP,
  switch: ONE_STEP,
  table: ONE_STEP,
  tabs: { sm: 'xl', md: '2xl', lg: '3xl' },
  tag: ONE_STEP,
  tagsInput: { xs: 'lg', sm: 'xl', md: '2xl', lg: '3xl' },
  textarea: TWO_STEPS,
};

const isFontKey = (key: string | undefined) => key === 'textStyle' || key === 'fontSize';

const FONT_TOKEN = /fontSizes\.(\w+)/g;

/** Rewrites text style and font size names, including `fontSizes.*` token references in CSS variables. */
export const renameFonts = (value: unknown, key?: string): unknown => {
  if (typeof value === 'string') {
    if (isFontKey(key) && Object.hasOwn(TYPE_NAMES, value)) {
      return TYPE_NAMES[value];
    }

    return value.replace(FONT_TOKEN, (match, name: string) =>
      Object.hasOwn(TYPE_NAMES, name) ? `fontSizes.${TYPE_NAMES[name]}` : match
    );
  }

  if (Array.isArray(value)) {
    return value.map((item) => renameFonts(item, key));
  }

  if (value && typeof value === 'object') {
    // Responsive and conditional font values keep their font key while descending.
    return Object.fromEntries(
      Object.entries(value).map(([entryKey, entry]) => [entryKey, renameFonts(entry, isFontKey(key) ? key : entryKey)])
    );
  }

  return value;
};

type AnyRecipe = RecipeDefinition | SlotRecipeDefinition;

/** Renames one recipe's fonts and, given a map, its size variants, default size, and compound size selections. */
export const rebaseRecipe = <T extends AnyRecipe>(name: string, recipe: T, sizes: NameMap | undefined): T => {
  const renamed = renameFonts(recipe) as T;

  if (!sizes || !renamed.variants?.size) {
    return renamed;
  }

  const rename = (size: string) => (SCALE_NAMES.has(size) ? sizes[size] : size);
  const sizeVariants: Record<string, unknown> = {};

  for (const [size, styles] of Object.entries(renamed.variants.size)) {
    const next = rename(size);

    if (next === undefined) {
      continue;
    }
    if (Object.hasOwn(sizeVariants, next)) {
      throw new Error(`Recipe ${name}: sizes collide on ${next}`);
    }
    sizeVariants[next] = styles;
  }

  const defaultSize = renamed.defaultVariants?.size;

  if (typeof defaultSize === 'string' && rename(defaultSize) === undefined) {
    throw new Error(`Recipe ${name}: default size ${defaultSize} has no workbench name`);
  }

  return {
    ...renamed,
    // Chakra iterates `compoundVariants` whenever the key exists, so absent stays absent.
    ...(renamed.compoundVariants && {
      compoundVariants: renamed.compoundVariants.map((compound: Record<string, unknown>) => {
        if (compound.size === undefined) {
          return compound;
        }
        const next = typeof compound.size === 'string' ? rename(compound.size) : undefined;
        if (next === undefined) {
          throw new Error(`Recipe ${name}: compound size ${String(compound.size)} has no workbench name`);
        }
        return { ...compound, size: next };
      }),
    }),
    defaultVariants:
      typeof defaultSize === 'string'
        ? { ...renamed.defaultVariants, size: rename(defaultSize) }
        : renamed.defaultVariants,
    variants: { ...renamed.variants, size: sizeVariants },
  };
};

/**
 * Rescaled recipes default to `md`, the working size, so a control that names no size is already dense. Heading
 * follows the type scale, and recipes without an `md` step or with a non-scale default (icon's `inherit`) keep theirs.
 */
const withWorkingDefault = <T extends AnyRecipe>(name: string, recipe: T): T =>
  name !== 'heading' &&
  SIZE_NAMES[name] &&
  recipe.variants?.size?.md &&
  SCALE_NAMES.has(String(recipe.defaultVariants?.size))
    ? { ...recipe, defaultVariants: { ...recipe.defaultVariants, size: 'md' } }
    : recipe;

const rebaseAll = <T extends AnyRecipe>(all: Record<string, T>): Record<string, T> =>
  Object.fromEntries(
    Object.entries(all).map(([name, recipe]) => [
      name,
      withWorkingDefault(name, rebaseRecipe(name, recipe, SIZE_NAMES[name])),
    ])
  );

export const recipes = rebaseAll<RecipeDefinition>(stockRecipes) as Record<keyof typeof stockRecipes, RecipeDefinition>;

export const slotRecipes = rebaseAll<SlotRecipeDefinition>(stockSlotRecipes) as Record<
  keyof typeof stockSlotRecipes,
  SlotRecipeDefinition
>;

const stockTheme = defaultConfig.theme ?? {};

/** Chakra's default config on the workbench scales; stock font names are replaced, not merged, so none survive. */
export const baseConfig: SystemConfig = {
  ...defaultConfig,
  theme: {
    ...stockTheme,
    recipes,
    slotRecipes,
    textStyles: {
      ...Object.fromEntries(
        Object.entries(TYPE_SCALE).map(([name, step]) => [name, { value: { ...step, fontSize: name } }])
      ),
      label: renameFonts(stockTheme.textStyles?.label) as NonNullable<typeof stockTheme.textStyles>[string],
      none: { value: {} },
    },
    tokens: {
      ...stockTheme.tokens,
      fontSizes: Object.fromEntries(Object.entries(TYPE_SCALE).map(([name, step]) => [name, { value: step.fontSize }])),
    },
  },
};
