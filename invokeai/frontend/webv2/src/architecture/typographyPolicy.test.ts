import { afterAll, describe, expect, it } from 'vitest';

import { isProductionSourcePath } from './dependencyPolicy';
import { closeSourceAnalysis, primeSourceAnalysis } from './tsSourceAnalysis';
import { checkTypographySource } from './typographyPolicy';

const sources = import.meta.glob('../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

afterAll(closeSourceAnalysis);

describe('retired typography tokens', () => {
  it('reports both typography properties with source positions and the replacement', () => {
    expect(checkTypographySource('View.tsx', '<Text fontSize="2xs" />;\nconst recipe = { textStyle: "2xs" };')).toEqual(
      [
        'View.tsx:1:16: fontSize uses retired typography token "2xs"; use "xs" for 10px type.',
        'View.tsx:2:29: textStyle uses retired typography token "2xs"; use "xs" for 10px type.',
      ]
    );
  });

  it.each([
    ['JSX expression', '<Text fontSize={"2xs"} />'],
    ['template literal', '<Text textStyle={`2xs`} />'],
    ['escaped string', '<Text fontSize={"\\u0032xs"} />'],
    ['conditional branch', '<Text fontSize={compact ? "2xs" : "md"} />'],
    ['logical fallback', '<Text fontSize={size ?? "2xs"} />'],
    ['logical alternative', '<Text fontSize={size || "2xs"} />'],
    ['conditional value', '<Text textStyle={compact && "2xs"} />'],
    ['responsive array', '<Text fontSize={["md", null, "2xs"]} />'],
    ['responsive object', '<Text textStyle={{ base: "xs", md: { _dark: "2xs" } }} />'],
    ['type assertion', '<Text fontSize={("2xs" as string)} />'],
    ['satisfies expression', '<Text fontSize={"2xs" satisfies string} />'],
    ['recipe condition', 'defineRecipe({ base: { _hover: { fontSize: "2xs" } } })'],
    ['style prop', '<Text css={{ fontSize: { base: "md", lg: "2xs" } }} />'],
    ['quoted property', '({ "textStyle": compact ? "xs" : "2xs" })'],
    ['computed property', '({ ["fontSize"]: "2xs" })'],
    ['responsive spread', '({ fontSize: { ...{ base: "2xs" } } })'],
  ])('rejects a retired %s value', (_, expression) => {
    const violations = checkTypographySource('example.tsx', `const example = ${expression};`);
    expect(violations).toHaveLength(1);
    expect(violations[0]).toContain('uses retired typography token "2xs"');
  });

  it('allows current tokens, custom CSS, unrelated literals, and deliberate stock-name mapping data', () => {
    expect(
      checkTypographySource(
        'allowed.tsx',
        `
          // The stock fontSize="2xs" is now xs.
          const example = <Text
            fontSize={{ base: 'xs', sm: 'sm', md: 'md', lg: '1.25rem' }}
            textStyle={mode === '2xs' ? 'xs' : 'sm'}
            data-size="2xs"
            size="2xs"
          />;
          const styles = { fontSize: 'var(--custom-font-size)', nested: { textStyle: 'label' } };
          const computed = <Text fontSize={renameStockSize('2xs')} />;
          const stockToWorkbench = { '2xs': 'xs', xs: 'md', sm: 'lg' };
          const stockNames = new Set(['2xs', 'xs', 'sm']);
          const fontSize = 'stockLabel';
          const stockLabels = { [fontSize]: '2xs' };
          const description = 'fontSize: "2xs"';
          type Stock = { fontSize: '2xs' | 'xs' };
        `
      )
    ).toEqual([]);
  });

  it('keeps retired typography values out of production JSX and style objects', () => {
    const entries = Object.entries(sources)
      .map(([path, source]) => [path.replace(/^\.\.\//, ''), source] as const)
      .filter(([path]) => isProductionSourcePath(path));

    primeSourceAnalysis(entries);
    const violations = entries.flatMap(([path, source]) => checkTypographySource(path, source));
    expect(violations).toEqual([]);
  }, 30_000);
});
