import { defaultConfig } from '@chakra-ui/react';
import { recipes as stockRecipes, slotRecipes as stockSlotRecipes } from '@chakra-ui/react/theme';
import { describe, expect, it } from 'vitest';

import { baseConfig, rebaseRecipe, recipes, renameFonts, SIZE_NAMES, slotRecipes, TYPE_NAMES } from './rebase';
import { CONTROL_HEIGHT_PX, type ControlSize } from './scale';

const stockTheme = defaultConfig.theme!;
const theme = baseConfig.theme!;

type TextStyleValue = { fontSize: string; letterSpacing?: string; lineHeight: string };
type StyleRecord = Record<string, Record<string, string>>;
type AnyDefaults = { defaultVariants?: Record<string, unknown> };
type StockRecipe = { base?: StyleRecord; variants?: { size?: Record<string, StyleRecord> } };

const stockRecipe = (name: string): StockRecipe =>
  ((stockRecipes as Record<string, StockRecipe>)[name] ?? (stockSlotRecipes as Record<string, StockRecipe>)[name])!;

const fontSizeOf = (tokens: typeof stockTheme.tokens, name: string) =>
  (tokens!.fontSizes as Record<string, { value: string }>)[name]?.value;

const textStyleOf = (styles: typeof stockTheme.textStyles, name: string) =>
  (styles as Record<string, { value: TextStyleValue }>)[name]?.value;

describe('workbench type scale', () => {
  it.each(Object.entries(TYPE_NAMES))('renders stock %s at the pixels it had, under the name %s', (stock, renamed) => {
    const before = textStyleOf(stockTheme.textStyles, stock);
    const after = textStyleOf(theme.textStyles, renamed);

    expect(fontSizeOf(theme.tokens, renamed)).toBe(fontSizeOf(stockTheme.tokens, stock));
    expect(after?.lineHeight).toBe(before?.lineHeight);
    expect(after?.letterSpacing).toBe(before?.letterSpacing);
    expect(after?.fontSize).toBe(renamed);
  });

  it('retires the stock 2xs name and adds an 11px step', () => {
    expect(fontSizeOf(theme.tokens, '2xs')).toBeUndefined();
    expect(textStyleOf(theme.textStyles, '2xs')).toBeUndefined();
    expect(fontSizeOf(theme.tokens, 'sm')).toBe('0.6875rem');
  });

  it('keeps the label text style at its stock size', () => {
    expect(textStyleOf(theme.textStyles, 'label')).toMatchObject({ fontSize: 'lg', lineHeight: '1.25rem' });
  });
});

describe('renameFonts', () => {
  it('renames text style and font size values, including responsive and conditional ones', () => {
    expect(
      renameFonts({
        _hover: { fontSize: 'xs' },
        fontSize: { base: '2xs', md: 'sm' },
        textStyle: 'sm',
      })
    ).toEqual({ _hover: { fontSize: 'md' }, fontSize: { base: 'xs', md: 'lg' }, textStyle: 'lg' });
  });

  it('renames font size tokens referenced from CSS variables', () => {
    expect(renameFonts({ '--indicator-font-size': 'fontSizes.xs', '--label': '{fontSizes.2xs}' })).toEqual({
      '--indicator-font-size': 'fontSizes.md',
      '--label': '{fontSizes.xs}',
    });
  });

  it('leaves the same names alone outside font properties', () => {
    const styles = { borderRadius: 'sm', boxShadow: 'md', fontSize: 'inherit', maxW: 'xs', px: '2' };

    expect(renameFonts(styles)).toEqual(styles);
  });
});

describe('rebaseRecipe', () => {
  it('renames sizes and the default size, drops unmapped scale steps, and keeps other keys', () => {
    const recipe = rebaseRecipe(
      'test',
      {
        defaultVariants: { size: 'sm' },
        variants: { size: { '2xl': { h: '16' }, full: { h: 'full' }, sm: { h: '9', textStyle: 'sm' } } },
      },
      { sm: 'lg' }
    );

    expect(recipe.variants?.size).toEqual({ full: { h: 'full' }, lg: { h: '9', textStyle: 'lg' } });
    expect(recipe.defaultVariants).toEqual({ size: 'lg' });
  });

  it('renames compound variant sizes and adds no compound key a recipe lacked', () => {
    const sizes = { sm: { h: '9' } };

    expect(
      rebaseRecipe('test', { compoundVariants: [{ css: {}, size: 'sm' }], variants: { size: sizes } }, { sm: 'md' })
        .compoundVariants
    ).toEqual([{ css: {}, size: 'md' }]);
    expect(rebaseRecipe('test', { variants: { size: sizes } }, { sm: 'md' })).not.toHaveProperty('compoundVariants');
  });

  it('refuses maps that would merge two sizes or orphan the default', () => {
    const variants = { size: { sm: { h: '8' }, xs: { h: '7' } } };

    expect(() => rebaseRecipe('test', { variants }, { sm: 'md', xs: 'md' })).toThrow(/collide on md/);
    expect(() => rebaseRecipe('test', { defaultVariants: { size: 'sm' }, variants }, { xs: 'md' })).toThrow(
      /default size sm/
    );
    expect(() =>
      rebaseRecipe('test', { compoundVariants: [{ css: {}, size: ['sm', 'xs'] }], variants }, { sm: 'md', xs: 'sm' })
    ).toThrow(/compound size/);
  });

  it('moves stock button sizes onto the control scale by height', () => {
    const stock = stockRecipe('button').variants!.size!;
    const renamed = recipes.button.variants?.size;

    expect(renamed?.sm).toMatchObject({ h: stock['2xs']!.h, textStyle: 'md' });
    expect(renamed?.md).toMatchObject({ h: stock.xs!.h, textStyle: 'md' });
    expect(renamed?.['3xl']).toMatchObject({ h: stock.lg!.h, textStyle: 'xl' });
    expect(Object.keys(renamed ?? {})).not.toContain('2xs');
  });

  // The theme pins button, input, select, combobox, and segment heights itself; these keep their stock heights, so
  // their names must land on the scale step of the same height.
  it.each([
    ['avatar', 'root', '--avatar-size'],
    ['colorPicker', 'trigger', '--input-height'],
    ['inputAddon', undefined, '--input-height'],
    ['nativeSelect', 'root', '--select-field-height'],
    ['tagsInput', 'root', '--tags-input-height'],
  ] as const)('names stock %s sizes by their pixel height', (name, part, variable) => {
    const stock = stockRecipe(name).variants!.size!;
    const sizes = stockTheme.tokens?.sizes as Record<string, { value: string }>;

    for (const [stockSize, renamed] of Object.entries(SIZE_NAMES[name]!)) {
      const styles = stock[stockSize];

      if (!styles) {
        continue;
      }
      const token = (part ? styles[part]?.[variable] : (styles as unknown as Record<string, string>)[variable])!;
      const px = Number.parseFloat(sizes[token.replace('sizes.', '')]!.value) * 16;

      expect(px, `${name} ${stockSize}`).toBe(CONTROL_HEIGHT_PX[renamed as ControlSize]);
    }
  });

  it('defaults rescaled recipes to the working size', () => {
    for (const name of ['badge', 'button', 'input', 'select', 'spinner', 'switch'] as const) {
      expect(
        (recipes as Record<string, AnyDefaults>)[name] ?? (slotRecipes as Record<string, AnyDefaults>)[name],
        name
      ).toMatchObject({ defaultVariants: { size: 'md' } });
    }
    // Headings follow the type scale; icons inherit their font size; tabs have no md step.
    expect(recipes.heading.defaultVariants).toMatchObject({ size: '3xl' });
    expect(recipes.icon.defaultVariants).toMatchObject({ size: 'inherit' });
    expect(slotRecipes.tabs.defaultVariants).toMatchObject({ size: '2xl' });
  });

  it('renames fonts in stock recipes it does not resize', () => {
    expect(slotRecipes.tooltip.base?.content).toMatchObject({ textStyle: 'md' });
    expect(stockRecipe('tooltip').base?.content).toMatchObject({ textStyle: 'xs' });
  });
});
