import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';

describe('architecture workflow', () => {
  it('confines unstable TypeScript imports to the source-analysis adapter', () => {
    const modules = import.meta.glob(['../**/*.{ts,tsx}', '../../scripts/**/*.{js,mjs,cjs,ts,mts,cts}'], {
      eager: true,
      import: 'default',
      query: '?raw',
    }) as Record<string, string>;
    const unstablePrefix = ['typescript', 'unstable'].join('/');
    const offenders = Object.entries(modules)
      .filter(
        ([path, source]) =>
          (!path.endsWith('/tsSourceAnalysis.ts') && source.includes(unstablePrefix)) ||
          (path.includes('/scripts/') && source.includes('./parse-source.mjs'))
      )
      .map(([path]) => path)
      .sort();
    expect(offenders).toEqual([]);
  });

  it('runs the release gates with base comparisons and retains diagnostics even on failure', () => {
    const workflows = import.meta.glob('../../../../../.github/workflows/frontend-tests.yml', {
      eager: true,
      import: 'default',
      query: '?raw',
    }) as Record<string, string>;
    const workflow = Object.values(workflows)[0] ?? '';

    expect(workflow).toContain('name: webv2-architecture-review');
    expect(workflow).toContain('invokeai/frontend/webv2/artifacts/architecture');
    expect(workflow).toContain('invokeai/frontend/webv2/artifacts/architecture-performance');
    expect(workflow).toContain('invokeai/frontend/webv2/artifacts/accessibility');
    expect(workflow).toContain('if: ${{ always()');

    const jobs = parse(workflow).jobs;
    const completion = jobs['frontend-webv2-tests'].steps.find((step: { run?: string }) =>
      step.run?.startsWith('pnpm check:')
    );
    expect(completion.run).toBe('pnpm check:release');
    expect(completion.env.WEBV2_PERF_REFERENCE_DIR).toContain('steps.perf-ref.outputs.cache-matched-key');
  });

  it('installs dependencies in the package that owns each frontend and generation gate', () => {
    const sources = import.meta.glob(
      [
        '../../../../../.github/workflows/{frontend-checks,frontend-tests,openapi-checks,typegen-checks}.yml',
        '../../../../../.github/actions/install-frontend-deps/action.yml',
      ],
      { eager: true, import: 'default', query: '?raw' }
    ) as Record<string, string>;
    const action = parse(Object.entries(sources).find(([path]) => path.endsWith('/action.yml'))![1]);
    expect(action.inputs['working-directory'].default).toBe('invokeai/frontend/webv2');
    for (const [path, source] of Object.entries(sources).filter(([path]) => path.includes('/workflows/'))) {
      const jobs = parse(source).jobs;
      for (const job of Object.values(jobs) as {
        defaults?: { run?: { 'working-directory'?: string } };
        steps: { uses?: string; with?: { 'working-directory'?: string } }[];
      }[]) {
        const install = job.steps.find((step) => step.uses === './.github/actions/install-frontend-deps');
        if (!install) {
          continue;
        }
        const expected =
          path.endsWith('/frontend-checks.yml') || path.endsWith('/frontend-tests.yml')
            ? 'invokeai/frontend/webv2'
            : 'invokeai/frontend/api';
        expect(install.with?.['working-directory'] ?? action.inputs['working-directory'].default, path).toBe(expected);
        if (job.defaults?.run) {
          expect(job.defaults.run['working-directory'], path).toBe(expected);
        }
      }
    }
  });

  it('runs every frontend-dependent gate when the shared Node version changes', () => {
    const sources = import.meta.glob(
      [
        '../../../../../.github/workflows/{frontend-checks,frontend-tests,openapi-checks,typegen-checks}.yml',
        '../../../../../.github/actions/install-frontend-deps/action.yml',
      ],
      { eager: true, import: 'default', query: '?raw' }
    ) as Record<string, string>;
    const action = parse(Object.entries(sources).find(([path]) => path.endsWith('/action.yml'))![1]);
    const nodeVersionFile = action.runs.steps.find((step: { uses?: string }) =>
      step.uses?.startsWith('actions/setup-node@')
    ).with['node-version-file'];
    const workflows = Object.entries(sources).filter(([path]) => path.includes('/workflows/'));
    expect(workflows).toHaveLength(4);
    for (const [path, source] of workflows) {
      const jobs = parse(source).jobs as Record<string, { steps: { uses?: string; with?: { files_yaml?: string } }[] }>;
      for (const job of Object.values(jobs)) {
        if (!job.steps.some((step) => step.uses === './.github/actions/install-frontend-deps')) {
          continue;
        }
        const filter = job.steps.find((step) => step.with?.files_yaml)?.with?.files_yaml ?? '';
        const patterns = Object.values(parse(filter) as Record<string, string[]>).flat();
        expect(patterns, path).toContain(nodeVersionFile);
      }
    }
  });
});
