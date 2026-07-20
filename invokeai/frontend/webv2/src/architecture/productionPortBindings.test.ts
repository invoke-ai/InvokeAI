import { productionPortBindingManifest } from '@app/productionPortBindings';
import { describe, expect, it } from 'vitest';

describe('production port binding composition', () => {
  it('enumerates the complete production binding manifest', () => {
    expect(productionPortBindingManifest).toEqual([
      { id: 'platform.http-auth', lifetime: 'pre-react', owner: 'platform' },
      { id: 'platform.authenticated-socket', lifetime: 'authenticated-session', owner: 'platform' },
      { id: 'queue.workbench-runtime', lifetime: 'editor', owner: 'queue' },
      { id: 'gallery.image-actions-bridge', lifetime: 'lazy-editor', owner: 'gallery' },
      { id: 'models.ui', lifetime: 'editor', owner: 'models' },
      { id: 'queue.ui', lifetime: 'editor', owner: 'queue' },
      { id: 'gallery.ui', lifetime: 'editor', owner: 'gallery' },
      { id: 'generation.ui', lifetime: 'editor', owner: 'generation' },
      { id: 'upscale.ui', lifetime: 'editor', owner: 'upscale' },
      { id: 'workflow.ui', lifetime: 'editor', owner: 'workflow' },
    ]);
  });

  it('has exactly one production mount declaration for every binding', () => {
    const appModules = import.meta.glob('../app/*.{ts,tsx}', {
      eager: true,
      import: 'default',
      query: '?raw',
    }) as Record<string, string>;
    const source = Object.entries(appModules)
      .filter(([fileName]) => !fileName.endsWith('/productionPortBindings.ts'))
      .map(([, contents]) => contents)
      .join('\n');

    for (const binding of productionPortBindingManifest) {
      const escapedId = binding.id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const mounts = source.match(
        new RegExp(`(?:bindProductionPortBinding|mountProductionPortBinding)\\(['"]${escapedId}['"]`, 'g')
      );
      expect(mounts, binding.id).toHaveLength(1);
    }
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
