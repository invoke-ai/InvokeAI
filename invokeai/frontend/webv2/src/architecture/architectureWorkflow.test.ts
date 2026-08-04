import { describe, expect, it } from 'vitest';

describe('architecture workflow', () => {
  it('confines unstable TypeScript imports to the source-analysis adapter', () => {
    const modules = import.meta.glob(
      [
        './*.ts',
        '../../scripts/{check-architecture-performance,write-canvas-import-matrix,write-contract-caller-matrix,write-workbench-ownership-inventory,parse-source}.mjs',
      ],
      { eager: true, import: 'default', query: '?raw' }
    ) as Record<string, string>;
    const unstablePrefix = ['typescript', 'unstable'].join('/');
    const offenders = Object.entries(modules)
      .filter(([path]) => !path.endsWith('/tsSourceAnalysis.ts'))
      .filter(([, source]) => source.includes(unstablePrefix) || source.includes('./parse-source.mjs'))
      .map(([path]) => path)
      .sort();
    expect(offenders).toEqual([]);
  });

  it('retains architecture inventories and performance reports in one CI artifact even on failure', () => {
    const workflows = import.meta.glob('../../../../../.github/workflows/frontend-tests.yml', {
      eager: true,
      import: 'default',
      query: '?raw',
    }) as Record<string, string>;
    const workflow = Object.values(workflows)[0] ?? '';

    expect(workflow).toContain('name: webv2-architecture-review');
    expect(workflow).toContain('invokeai/frontend/webv2/artifacts/architecture');
    expect(workflow).toContain('invokeai/frontend/webv2/artifacts/architecture-performance');
    expect(workflow).toContain('if: ${{ always()');
  });
});
