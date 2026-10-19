import { afterAll, describe, expect, it } from 'vitest';

import {
  checkDependency,
  checkSource,
  collectImportReferences,
  formatViolation,
  getModuleOwner,
  isExcepted,
  isProductionSourcePath,
  primeImportSources,
  resolveImportPath,
} from './dependencyPolicy';
import { FEATURE_PUBLIC_INTERFACES } from './featureInterfaces';
import { migrationExceptions } from './migrationExceptions';
import { closeSourceAnalysis } from './tsSourceAnalysis';

const sources = import.meta.glob('../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

const toSourcePath = (path: string): string => path.replace(/^\.\.\//, '');
const isProduction = isProductionSourcePath;

afterAll(closeSourceAnalysis);

describe('dependency policy parser', () => {
  it('classifies every supported TypeScript import form', () => {
    const references = collectImportReferences(`
      import value from '@platform/value';
      import type { Type } from '@platform/type';
      export { named } from '@platform/named';
      export type { NamedType } from '@platform/named-type';
      export * from '@platform/star';
      export * as ns from '@platform/namespace';
      type Inline = import('@platform/inline').Inline;
      const lazy = import('@features/gallery');
    `);

    expect(references).toEqual([
      { exposesCanvasEngine: false, kind: 'import', specifier: '@platform/value' },
      { exposesCanvasEngine: false, kind: 'import-type', specifier: '@platform/type' },
      { exposesCanvasEngine: false, kind: 'export', specifier: '@platform/named' },
      { exposesCanvasEngine: false, kind: 'export', specifier: '@platform/named-type' },
      { exposesCanvasEngine: true, kind: 'export-star', specifier: '@platform/star' },
      { exposesCanvasEngine: true, kind: 'export-star', specifier: '@platform/namespace' },
      { exposesCanvasEngine: false, kind: 'import-type', specifier: '@platform/inline' },
      { exposesCanvasEngine: false, kind: 'dynamic-import', specifier: '@features/gallery' },
    ]);
  });

  it('normalizes aliases and relative paths before assigning owners', () => {
    expect(resolveImportPath('features/queue/ui/View.tsx', '../../gallery')).toBe('features/gallery');
    expect(resolveImportPath('features/queue/ui/View.tsx', '@workbench/canvas-engine/api')).toBe(
      'workbench/canvas-engine/api'
    );
    expect(getModuleOwner('features/queue/index.ts')).toBe('feature:queue');
    expect(getModuleOwner('platform/query/client.ts')).toBe('platform');
  });
});

describe('dependency policy rules', () => {
  it('rejects Queue-to-Workbench dependencies with an actionable message', () => {
    const violations = checkDependency('features/queue/runtime.ts', '@workbench/workbenchStore');

    expect(violations.map(formatViolation)).toEqual([
      'feature-dependency-direction: features/queue/runtime.ts -> workbench/workbenchStore',
    ]);
  });

  it('rejects cross-feature implementation bypasses but permits public interfaces', () => {
    expect(checkDependency('features/queue/ui/View.tsx', '@features/gallery/data/api')).toMatchObject([
      { rule: 'feature-public-interface' },
    ]);
    expect(checkDependency('features/queue/ui/View.tsx', '@features/gallery')).toEqual([]);
  });

  it('rejects private Canvas paths outside Canvas and permits its public API', () => {
    expect(checkDependency('workbench/widgets/canvas/View.tsx', '@workbench/canvas-engine/engine')).toMatchObject([
      { rule: 'canvas-private-interface' },
    ]);
    expect(checkDependency('workbench/widgets/canvas/View.tsx', '@workbench/canvas-engine/api')).toEqual([]);
    expect(checkDependency('workbench/widgets/canvas/View.tsx', '@workbench/canvas-operations/api')).toEqual([]);
    expect(checkDependency('workbench/widgets/canvas/View.tsx', '@workbench/canvas-operations/react')).toEqual([]);
  });

  it.each([
    `export type { CanvasEngine } from '@workbench/canvas-engine/engine';`,
    `export { CanvasEngine as PublicEngine } from '@workbench/canvas-engine/engine';`,
    `export * from '@workbench/canvas-engine/engine';`,
    `export * as Core from '@workbench/canvas-engine/engine';`,
    `import type * as Core from '@workbench/canvas-engine/engine'; type Consumer = Core.CanvasEngine;`,
    `type Consumer = import('@workbench/canvas-operations/createCanvasEngine').CanvasEngine;`,
  ])('keeps the full Canvas engine private: %s', (source) => {
    expect(checkSource('workbench/widgets/canvas/consumer.ts', source)).toContainEqual(
      expect.objectContaining({ rule: 'canvas-construction-private' })
    );
  });

  it('keeps Canvas engine implementation independent of application workflows', () => {
    expect(checkDependency('workbench/canvas-engine/engine.ts', '@features/generation/graph')).toContainEqual(
      expect.objectContaining({ rule: 'canvas-engine-independence' })
    );
  });

  it('permits same-module implementation imports', () => {
    expect(checkDependency('features/queue/ui/View.tsx', '../data/queries')).toEqual([]);
    expect(checkDependency('workbench/canvas-engine/controllers/a.ts', '../engine')).toEqual([]);
  });

  it.each([
    `import { apiFetchJson } from '@platform/transport/http';`,
    `import type { BackendConnectionStatus } from '@platform/transport/types';`,
    `import { Button } from '@platform/ui';`,
    `import type { ColorPalette } from '@chakra-ui/react';`,
    `import type { ReactFlowInstance } from '@xyflow/react';`,
    `import { ImageIcon } from 'lucide-react';`,
    `import type { ReactNode } from 'react';`,
    `type Button = import('@platform/ui').Button;`,
  ])('keeps feature Core independent of transport, React, and UI imports: %s', (source) => {
    expect(checkSource('features/example/core/policy.ts', source)).toContainEqual(
      expect.objectContaining({ rule: 'feature-core-purity' })
    );
  });

  it.each([
    'react',
    '@chakra-ui/react',
    'i18next',
    '@platform/ui',
    '@workbench/canvas-engine/render/compositor',
    '@workbench/canvas-engine/history/documentPatch',
    '@workbench/canvas-engine/engine',
    '@workbench/canvasProjectMutations',
    '@workbench/widgets/layers/layerOps',
    '@workbench/canvas-engine/document/sources',
    '@workbench/canvas-engine/document/bitmapStore',
  ])('keeps the pure document model free of engine, screen and framework imports: %s', (specifier) => {
    expect(checkDependency('workbench/canvas-engine/document-model/documentModel.ts', specifier)).toContainEqual(
      expect.objectContaining({ rule: 'document-model-purity' })
    );
  });

  it('lets the reducer share the document seam helpers without widening the public API', () => {
    const reducer = 'workbench/canvasProjectMutations.ts';
    expect(checkDependency(reducer, '@workbench/canvas-engine/document/layerStacks')).toEqual([]);
    expect(checkDependency(reducer, '@workbench/canvas-engine/document/selectionRepair')).toEqual([]);
    expect(checkDependency(reducer, '@workbench/canvas-engine/document/insertionAnchors')).toEqual([]);
    expect(checkDependency(reducer, '@workbench/canvas-engine/document/sources')).toContainEqual(
      expect.objectContaining({ rule: 'canvas-private-interface' })
    );
    expect(
      checkDependency('workbench/widgets/layers/layerGroups.ts', '@workbench/canvas-engine/document/layerStacks')
    ).toContainEqual(expect.objectContaining({ rule: 'canvas-private-interface' }));
  });

  it('permits the pure document model to read document facts, math and contracts', () => {
    const source = 'workbench/canvas-engine/document-model/documentModel.ts';
    expect(checkDependency(source, './semanticLeaf')).toEqual([]);
    expect(checkDependency(source, '@workbench/canvas-engine/document/layerStacks')).toEqual([]);
    expect(checkDependency(source, '@workbench/canvas-engine/math/mat2d')).toEqual([]);
    expect(checkDependency(source, '@workbench/canvas-engine/contracts')).toEqual([]);
    expect(checkDependency(source, '@workbench/canvas-engine/mutationContracts')).toEqual([]);
    expect(checkDependency('workbench/canvas-engine/document-model/documentModel.test.ts', 'vitest')).toEqual([]);
  });

  it('permits feature Core to depend on other pure Core and Platform State modules', () => {
    expect(checkDependency('features/queue/core/policy.ts', './types')).toEqual([]);
    expect(checkDependency('features/queue/core/policy.ts', '@platform/state/selectorCore')).toEqual([]);
  });

  it('treats test-support modules as non-production and keeps production out of them', () => {
    expect(isProductionSourcePath('features/generation/core/architectureCapabilities.testing.ts')).toBe(false);
    expect(isProductionSourcePath('platform/ui/Button.type-test.tsx')).toBe(false);
    expect(isProductionSourcePath('features/generation/ui/GenerateWidgetView.browser.test.tsx')).toBe(false);
    expect(isProductionSourcePath('features/generation/core/architectureCapabilities.ts')).toBe(true);
    // The helper imports a test runner and a 1000-row fixture, and Core is bundled by every route.
    expect(
      checkDependency('features/generation/ui/GenerateWidgetView.tsx', '../core/architectureCapabilities.testing')
    ).toContainEqual(expect.objectContaining({ rule: 'test-support-isolation' }));
    expect(checkDependency('features/generation/core/settings.test.ts', './architectureCapabilities.testing')).toEqual(
      []
    );
  });

  it('rejects the retired global Workbench contract hub', () => {
    expect(checkDependency('workbench/shell/View.tsx', '@workbench/types')).toContainEqual(
      expect.objectContaining({ rule: 'retired-contract-hub' })
    );
  });

  it('rejects private feature paths and broad star exports across ownership seams', () => {
    expect(checkDependency('workbench/backend/coordinator.ts', '@features/queue/core/types')).toContainEqual(
      expect.objectContaining({ rule: 'feature-private-interface' })
    );
    expect(checkDependency('workbench/projectContracts.ts', '@features/queue/contracts')).toEqual([]);
    expect(checkDependency('workbench/backend/coordinator.ts', '@features/queue')).toEqual([]);
    expect(checkSource('workbench/queue.ts', `export * from '@features/queue';`)).toContainEqual(
      expect.objectContaining({ rule: 'cross-owner-star-export' })
    );
    expect(checkSource('platform/ui/index.ts', `export * from './Button';`)).toEqual([]);
  });

  it('keeps fixtures subject to the same parser and rules', () => {
    expect(checkSource('platform/query/client.ts', `export * from '@features/queue';`)).toEqual(
      expect.arrayContaining([expect.objectContaining({ rule: 'platform-independence' })])
    );
    expect(
      checkSource('features/queue/runtime.ts', `type Store = import('@workbench/workbenchStore').WorkbenchStore;`)
    ).toMatchObject([{ rule: 'feature-dependency-direction' }]);
  });
});

describe('feature public-interface registry', () => {
  it('rejects top-level modules a feature has not registered', () => {
    expect(checkDependency('workbench/shell/View.tsx', '@features/workflow/data')).toMatchObject([
      { rule: 'feature-private-interface' },
    ]);
    expect(checkDependency('workbench/shell/View.tsx', '@features/identity/session')).toMatchObject([
      { rule: 'feature-private-interface' },
    ]);
    expect(checkDependency('workbench/shell/View.tsx', '@features/identity')).toEqual([]);
    expect(checkDependency('workbench/palette/paletteProviders.ts', '@features/workflow/queries')).toEqual([]);
    expect(checkDependency('workbench/palette/paletteProviders.ts', '@features/workflow/paletteSearch')).toEqual([]);
    expect(checkDependency('workbench/palette/paletteProviders.ts', '@features/gallery/paletteSearch')).toEqual([]);
    expect(checkDependency('workbench/palette/paletteProviders.ts', '@features/queue/queries')).toEqual([]);
    expect(checkDependency('workbench/palette/WorkbenchCommandPaletteDialog.tsx', '@features/queue/queries')).toEqual(
      []
    );
    expect(checkDependency('workbench/palette/WorkbenchCommandPalette.tsx', '@features/queue/reveal')).toEqual([]);
    expect(checkDependency('features/gallery/ui/View.tsx', '@features/queue/publicApi')).toMatchObject([
      { rule: 'feature-public-interface' },
    ]);
    expect(checkDependency('workbench/invocationSubmit.ts', '@features/generation/prompts')).toEqual([]);
    expect(checkDependency('app/GenerateWidgetSyncRuntime.tsx', '@features/generation/runtime')).toEqual([]);
    expect(checkDependency('workbench/widgets/canvas/invoke.ts', '@features/generation/canvasGraph')).toEqual([]);
    expect(checkDependency('workbench/WorkbenchContext.tsx', '@features/intermediates/holdLease')).toEqual([]);
    expect(
      checkDependency('features/workflow/ui/fields/RecordPickerInput.tsx', '@features/generation/systemPrompts')
    ).toEqual([]);
    expect(checkDependency('workbench/invocationSubmit.ts', '@features/workflow/generators')).toEqual([]);
    expect(
      checkDependency('workbench/invocationSubmit.ts', '@features/generation/data/dynamicPromptsQueries')
    ).toMatchObject([{ rule: 'feature-private-interface' }]);
    expect(
      checkDependency('features/workflow/ui/fields/RecordPickerInput.tsx', '@features/generation/data/systemPrompts')
    ).toMatchObject([{ rule: 'feature-public-interface' }]);
  });

  it('treats unregistered features as fully private', () => {
    expect(checkDependency('workbench/shell/View.tsx', '@features/imaginary')).toMatchObject([
      { rule: 'feature-private-interface' },
    ]);
    expect(checkDependency('workbench/shell/View.tsx', '@features/imaginary/contracts')).toMatchObject([
      { rule: 'feature-private-interface' },
    ]);
  });

  it('registers only real features and existing entry modules', () => {
    const productionPaths = new Set(Object.keys(sources).map(toSourcePath));
    for (const [feature, modules] of Object.entries(FEATURE_PUBLIC_INTERFACES)) {
      expect(
        [...productionPaths].some((path) => path.startsWith(`features/${feature}/`)),
        `features/${feature} does not exist`
      ).toBe(true);
      for (const module of modules) {
        expect(
          productionPaths.has(`features/${feature}/${module}.ts`) ||
            productionPaths.has(`features/${feature}/${module}.tsx`),
          `features/${feature}/${module} does not exist`
        ).toBe(true);
      }
    }
  });
});

describe('production dependency graph', () => {
  it('classifies every production import and has no unowned violation', () => {
    primeImportSources(Object.entries(sources).map(([path, source]) => [toSourcePath(path), source] as const));

    const violations = Object.entries(sources)
      .filter(([path]) => isProduction(path))
      .flatMap(([path, source]) => checkSource(toSourcePath(path), source))
      .filter((violation) => !migrationExceptions.some((exception) => isExcepted(violation, exception)))
      .map(formatViolation)
      .sort();

    expect(violations).toEqual([]);
    // A whole-source scan whose cost grows with the codebase; on shared CI runners it already takes 3-5s, so the
    // default 5s budget turned runner speed into test failures.
  }, 60_000);

  it('has no remaining migration exceptions', () => {
    expect(migrationExceptions).toEqual([]);
  });
});
