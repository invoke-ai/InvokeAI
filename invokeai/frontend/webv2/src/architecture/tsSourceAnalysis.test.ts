import { afterEach, describe, expect, it } from 'vitest';

import { analyzeSource, closeSourceAnalysis, primeSourceAnalysis } from './tsSourceAnalysis';

afterEach(() => {
  closeSourceAnalysis();
});

describe('tsSourceAnalysis', () => {
  it('classifies declaration, inline, and dynamic module references', () => {
    const analysis = analyzeSource(
      'forms.ts',
      `
        import value from 'value';
        import type { ExternalType as LocalType } from 'types';
        import defer * as deferred from 'deferred';
        import 'side-effect';
        export { external as local } from 'named-export';
        export * from 'star-export';
        export * as namespace from 'namespace-export';
        type Inline = import('inline').Inline;
        const lazy = import('dynamic');
      `
    );

    expect(analysis.moduleReferences).toEqual([
      {
        form: 'import-declaration',
        kind: 'import',
        namespace: false,
        qualifier: null,
        specifier: 'value',
        symbols: ['default'],
      },
      {
        form: 'import-declaration',
        kind: 'import-type',
        namespace: false,
        qualifier: null,
        specifier: 'types',
        symbols: ['ExternalType'],
      },
      {
        form: 'import-declaration',
        kind: 'import',
        namespace: true,
        qualifier: null,
        specifier: 'deferred',
        symbols: ['*'],
      },
      {
        form: 'import-declaration',
        kind: 'import',
        namespace: false,
        qualifier: null,
        specifier: 'side-effect',
        symbols: [],
      },
      {
        form: 'export-declaration',
        kind: 'export',
        namespace: false,
        qualifier: null,
        specifier: 'named-export',
        symbols: ['external'],
      },
      {
        form: 'export-declaration',
        kind: 'export-star',
        namespace: true,
        qualifier: null,
        specifier: 'star-export',
        symbols: ['*'],
      },
      {
        form: 'export-declaration',
        kind: 'export-star',
        namespace: true,
        qualifier: null,
        specifier: 'namespace-export',
        symbols: ['*'],
      },
      {
        form: 'import-type',
        kind: 'import-type',
        namespace: false,
        qualifier: 'Inline',
        specifier: 'inline',
        symbols: [],
      },
      {
        form: 'dynamic-import',
        kind: 'dynamic-import',
        namespace: false,
        qualifier: null,
        specifier: 'dynamic',
        symbols: [],
      },
    ]);
  });

  it('collects public exports', () => {
    expect(
      analyzeSource(
        'exports.ts',
        `
          export class PublicClass {}
          class PrivateClass {}
          export const publicValue = 1;
          const privateValue = 2;
          export default privateValue;
          export { PrivateClass as Renamed };
          export * from 'external';
        `
      ).publicExports
    ).toEqual(['*', 'PublicClass', 'Renamed', 'default', 'publicValue']);
  });

  it('identifies type-only modules', () => {
    expect(
      analyzeSource(
        'types.ts',
        `
          import type { Input } from 'input';
          export type { Output } from 'output';
          interface Shape { value: Input }
          type Alias = Shape;
        `
      ).typeOnly
    ).toBe(true);
  });

  it('does not identify deferred imports as type-only', () => {
    expect(analyzeSource('deferred.ts', `import defer * as runtime from 'runtime';`).typeOnly).toBe(false);
  });

  it('replaces stale source text and supports idempotent teardown', () => {
    primeSourceAnalysis([['same.ts', "import 'first';"]]);
    expect(analyzeSource('same.ts', "import 'first';").moduleReferences[0]?.specifier).toBe('first');
    expect(analyzeSource('same.ts', "import 'second';").moduleReferences[0]?.specifier).toBe('second');

    closeSourceAnalysis();
    closeSourceAnalysis();

    expect(analyzeSource('same.ts', "import 'third';").moduleReferences[0]?.specifier).toBe('third');
  });
});
