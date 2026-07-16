import { describe, expect, it } from 'vitest';

import { referencesFullCanvasEngine } from './fullEngineImportAudit';

const modules = import.meta.glob('../**/*.{ts,tsx}', { eager: true, import: 'default', query: '?raw' });

const ALLOWED = new Set(['./createCanvasEngine.ts', './engineRegistry.ts']);
const fullEngineImports = (): string[] => {
  const result: string[] = [];
  for (const [file, source] of Object.entries(modules)) {
    const relative = file.replace('../', '');
    if (
      relative.endsWith('.test.ts') ||
      relative.endsWith('.test.tsx') ||
      ALLOWED.has(relative) ||
      typeof source !== 'string'
    ) {
      continue;
    }
    if (referencesFullCanvasEngine(source)) {
      result.push(relative);
    }
  }
  return result.sort();
};

describe('full CanvasEngine consumer boundary', () => {
  it.each([
    [`export type { CanvasEngine } from '@workbench/canvas-engine/engine';`, 'named type re-export'],
    [`export { CanvasEngine as PublicEngine } from '@workbench/canvas-engine/engine';`, 'aliased re-export'],
    [`export * from '@workbench/canvas-engine/engine';`, 'star re-export'],
    [`export * as Core from '@workbench/canvas-engine/engine';`, 'namespace re-export'],
    [
      `import type * as Core from '@workbench/canvas-engine/engine'; type Consumer = Core.CanvasEngine;`,
      'qualified namespace reference',
    ],
    [`import * as Core from '@workbench/canvas-engine/engine'; void Core;`, 'namespace import'],
    [`type Consumer = import('@workbench/canvas-operations/createCanvasEngine').CanvasEngine;`, 'inline imported type'],
  ])('detects $1', (source) => {
    expect(referencesFullCanvasEngine(source)).toBe(true);
  });

  it('does not reject a named import of a narrow capability', () => {
    expect(
      referencesFullCanvasEngine(
        `import type { CanvasCoreStores } from '@workbench/canvas-engine/engine'; type Stores = CanvasCoreStores;`
      )
    ).toBe(false);
  });

  it('limits the full engine type to registry and composition roots', () => {
    expect(fullEngineImports()).toEqual([]);
  });
});
