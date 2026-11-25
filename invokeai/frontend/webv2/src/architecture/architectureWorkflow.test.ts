import { afterAll, describe, expect, it } from 'vitest';
import { parse } from 'yaml';

import { collectImportReferences, isProductionSourcePath } from './dependencyPolicy';
import { closeSourceAnalysis } from './tsSourceAnalysis';

interface Step {
  'continue-on-error'?: unknown;
  env?: Record<string, string>;
  id?: string;
  if?: string;
  name?: string;
  run?: string;
  uses?: string;
  with?: Record<string, unknown>;
  'working-directory'?: string;
}

interface Job {
  'continue-on-error'?: unknown;
  if?: string;
  needs?: string | string[];
  steps: Step[];
  strategy?: { matrix?: Record<string, unknown> };
}

const sources = import.meta.glob(
  ['../../package.json', '../../../../../.github/workflows/{frontend-checks,frontend-tests}.yml'],
  { eager: true, import: 'default', query: '?raw' }
) as Record<string, string>;
const packageSource = sources['../../package.json']!;
const parseWorkflow = (name: 'frontend-checks' | 'frontend-tests') =>
  parse(sources[`../../../../../.github/workflows/${name}.yml`]!) as {
    jobs: unknown;
    on: unknown;
    permissions: unknown;
  };

const ownFilterPatterns = (job: Job): string[] => {
  const filter = job.steps.find((step) => step.with?.files_yaml)?.with?.files_yaml;
  return typeof filter === 'string' ? Object.values(parse(filter) as Record<string, string[]>).flat() : [];
};

/** The change filter that gates a job: its own, or that of the jobs it needs. */
const changeFilterPatterns = (jobs: Record<string, Job>, name: string): string[] => {
  const job = jobs[name]!;
  const needs = typeof job.needs === 'string' ? [job.needs] : (job.needs ?? []);
  return [...ownFilterPatterns(job), ...needs.flatMap((need) => ownFilterPatterns(jobs[need]!))];
};

/** A GitHub Actions expression, `${{ body }}`. */
const expression = (body: string): string => `\${{ ${body} }}`;

afterAll(closeSourceAnalysis);

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

  it('requires every test job through the always-running status check and keeps their diagnostics', () => {
    const jobs = parseWorkflow('frontend-tests').jobs as Record<string, Job>;
    const required = jobs['frontend-webv2-tests']!;
    // A skipped job satisfies a required check: the check must run after failures and cancellations too.
    expect(required.if).toBe(expression('always()'));
    expect([...(required.needs ?? [])].sort()).toEqual(
      Object.keys(jobs)
        .filter((name) => name !== 'frontend-webv2-tests')
        .sort()
    );
    // A job that continues on error reports success to the required check.
    expect(Object.entries(jobs).filter(([, job]) => job['continue-on-error'] !== undefined)).toEqual([]);
    const steps = Object.values(jobs).flatMap((job) => job.steps);
    expect(steps.filter((step) => step['continue-on-error'] !== undefined)).toEqual([]);
    const uploads = steps.filter((step) => step.uses?.startsWith('actions/upload-artifact@'));
    for (const upload of uploads) {
      expect(upload.if).toContain('always()');
    }
    const uploaded = uploads.flatMap((upload) => String(upload.with?.path).trim().split('\n'));
    for (const output of ['architecture-performance', 'accessibility', 'browser-smoke']) {
      expect(uploaded).toContain(`invokeai/frontend/webv2/artifacts/${output}`);
    }
  });

  it('compares with a reference keyed on every input of the exact base measurement', () => {
    const workflow = parseWorkflow('frontend-tests');
    const jobs = workflow.jobs as Record<string, Job>;
    const steps = Object.values(jobs).find((job) =>
      job.steps.some((step) => step.run === 'pnpm test:performance:architecture')
    )!.steps;
    const byId = (id: string) => steps.find((step) => step.id === id)!;
    const byName = (name: string) => steps.find((step) => step.name === name)!;

    const key = byId('perf-key');
    expect(key.env?.REFERENCE_COMMIT).toBe(
      expression('github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.sha')
    );
    const inputs = /inputs=\(([^)]*)\)/
      .exec(key.run ?? '')![1]!
      .trim()
      .split(/\s+/);
    expect(inputs).toEqual([
      'invokeai/frontend/webv2',
      'invokeai/version/invokeai_version.py',
      '.nvmrc',
      '.github/actions/install-frontend-deps',
      '.github/workflows/frontend-tests.yml',
    ]);
    // Commits that change an input must record a reference, so each input has to trigger this workflow.
    const patterns = changeFilterPatterns(jobs, 'performance');
    for (const input of inputs) {
      expect(patterns, input).toSatisfy((list: string[]) => list.includes(input) || list.includes(`${input}/**`));
    }
    // A prefix fallback could compare with another base's measurements.
    const cacheSteps = steps.filter((step) => step.uses?.startsWith('actions/cache/'));
    expect(cacheSteps.length).toBeGreaterThanOrEqual(3);
    for (const step of cacheSteps) {
      expect(step.with?.key).toBe(expression('steps.perf-key.outputs.key'));
      expect(step.with?.['restore-keys']).toBeUndefined();
    }
    expect(byName('performance gates').env?.WEBV2_PERF_REFERENCE_DIR).toContain("steps.perf-ref.outcome == 'success'");

    const checkout = byId('perf-base');
    expect(checkout.with?.['persist-credentials']).toBe(false);
    const install = byName('install performance base dependencies');
    expect(install.with?.['working-directory']).toBe(`${String(checkout.with?.path)}/invokeai/frontend/webv2`);
    expect(install.with?.['node-version-file']).toBe(`${String(checkout.with?.path)}/.nvmrc`);
    const measure = byId('perf-base-measure');
    expect(measure['working-directory']).toBe(install.with?.['working-directory']);
    expect(measure.env?.WEBV2_PERF_REFERENCE_DIR).toBe('');
    const candidateNode = byName('restore candidate Node version');
    expect(candidateNode.with?.['node-version-file']).toBe('.nvmrc');
    expect(steps.indexOf(candidateNode)).toBeGreaterThan(steps.indexOf(measure));
    expect(steps.indexOf(candidateNode)).toBeLessThan(steps.indexOf(byName('performance gates')));

    expect(workflow.permissions).toEqual({ contents: 'read' });
    for (const event of ['push', 'pull_request', 'merge_group', 'workflow_dispatch', 'workflow_call']) {
      expect(workflow.on).toHaveProperty(event);
    }
  });

  it('builds production code from type-only API imports, so the API package is no reference input', () => {
    const sources = import.meta.glob('../**/*.{ts,tsx}', { eager: true, import: 'default', query: '?raw' }) as Record<
      string,
      string
    >;
    const offenders = Object.entries(sources)
      .filter(([path, source]) => isProductionSourcePath(path) && /['"]@api[/'"]/.test(source))
      .flatMap(([path, source]) =>
        collectImportReferences(source, path)
          .filter(({ kind, specifier }) => /^@api(?:\/|$)/.test(specifier) && kind !== 'import-type')
          .map(({ specifier }) => `${path} imports ${specifier} as a value`)
      );
    expect(offenders).toEqual([]);
  });

  it('runs each completion gate once locally and across the frontend CI jobs', () => {
    const scripts = JSON.parse(packageSource).scripts as Record<string, string>;
    const expand = (command: string): string[] =>
      command.split(' && ').flatMap((part) => {
        const [, script, args] = /^pnpm (?:run )?([\w:-]+)((?: .+)?)$/.exec(part) ?? [];
        if (!script || !scripts[script]) {
          return [part];
        }
        // pnpm appends arguments to the script, so they reach its last command.
        const commands = expand(scripts[script]);
        return [...commands.slice(0, -1), `${commands.at(-1)!}${args}`];
      });
    const workflows = (['frontend-checks', 'frontend-tests'] as const).map(
      (name) => [name, parseWorkflow(name)] as const
    );
    const ciJobs = workflows.flatMap(([, workflow]) =>
      Object.values(workflow.jobs as Record<string, Job>).map((job) => ({
        job,
        commands: job.steps.flatMap((step) => (step.run ? expand(step.run) : [])),
      }))
    );
    const ci = ciJobs.flatMap(({ commands }) => commands);
    const local = expand('pnpm check:release');
    // CI must follow the package chain: every local gate runs in CI exactly as often as locally, which is once. The
    // build is the exception: each CI job that reads `dist` builds for itself (checked below).
    const localGates = local.filter((command) => command !== 'vite build');
    expect(new Set(localGates).size, 'local release repeats a gate').toBe(localGates.length);
    const sharded = / --shard=.+$/;
    const unsharded = (command: string) => command.replace(sharded, '');
    for (const gate of localGates) {
      expect(
        ci.filter((command) => unsharded(command) === gate),
        `CI runs ${gate}`
      ).toHaveLength(1);
    }
    // Architecture tests are part of the complete unit run, not a second targeted invocation.
    for (const commands of [local, ci]) {
      expect(commands.filter((command) => command.startsWith('vitest run --config vitest.config.mts'))).toHaveLength(1);
    }
    // A matrix repeats its gates once per leg; only the sharded browser suite may run in one.
    for (const { job, commands } of ciJobs) {
      if (job.strategy?.matrix && commands.some((command) => localGates.includes(unsharded(command)))) {
        expect(commands.filter((command) => localGates.includes(unsharded(command)))).toSatisfy((gates: string[]) =>
          gates.every((command) => sharded.test(command))
        );
      }
    }
    // Locally the performance build serves every later journey; each CI job that runs one builds first.
    expect(local.filter((command) => command === 'vite build')).toHaveLength(1);
    const builtOutput = /scripts\/(?:check-architecture-performance|measure-architecture-performance|run-[\w-]+)\.mjs$/;
    for (const { commands } of ciJobs) {
      const firstReader = commands.findIndex((command) => builtOutput.test(command));
      if (firstReader !== -1) {
        expect(commands.slice(0, firstReader)).toContain('vite build');
      }
    }
    // A sharded suite runs every file only if the shards are 1..N of N jobs with no other matrix dimension.
    for (const { job, commands } of ciJobs) {
      const shard = commands.map((command) => / --shard=(.+)$/.exec(command)?.[1]).find(Boolean);
      if (shard === undefined) {
        continue;
      }
      expect(shard).toBe(`${expression('matrix.shard')}/${expression('strategy.job-total')}`);
      const matrix = job.strategy?.matrix ?? {};
      expect(Object.keys(matrix)).toEqual(['shard']);
      expect(matrix.shard).toEqual(Array.from({ length: (matrix.shard as unknown[]).length }, (_, index) => index + 1));
    }
    for (const [name, workflow] of workflows) {
      const patterns = Object.values(workflow.jobs as Record<string, Job>).flatMap((job) => ownFilterPatterns(job));
      for (const [other] of workflows) {
        expect(patterns, name).toContain(`.github/workflows/${other}.yml`);
      }
      // A backend change regenerates the API types the type check reads, so it has to run the check too.
      if (name === 'frontend-checks') {
        expect(patterns, name).toContain('invokeai/frontend/api/schema.ts');
      }
    }
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
    const nodeVersionInput = action.runs.steps.find((step: { uses?: string }) =>
      step.uses?.startsWith('actions/setup-node@')
    ).with['node-version-file'];
    expect(nodeVersionInput).toMatch(/^\$\{\{ inputs.node-version-file \}\}$/);
    const nodeVersionFile = action.inputs['node-version-file'].default;
    expect(nodeVersionFile).toBe('.nvmrc');
    const workflows = Object.entries(sources).filter(([path]) => path.includes('/workflows/'));
    expect(workflows).toHaveLength(4);
    for (const [path, source] of workflows) {
      const jobs = parse(source).jobs as Record<string, Job>;
      for (const [name, job] of Object.entries(jobs)) {
        if (!job.steps.some((step) => step.uses === './.github/actions/install-frontend-deps')) {
          continue;
        }
        expect(changeFilterPatterns(jobs, name), `${path} ${name}`).toContain(nodeVersionFile);
      }
    }
  });
});
