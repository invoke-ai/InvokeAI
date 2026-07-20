import { describe, expect, it } from 'vitest';

describe('architecture workflow', () => {
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
