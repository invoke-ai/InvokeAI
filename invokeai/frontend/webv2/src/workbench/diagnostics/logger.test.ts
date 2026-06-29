import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  clearProjectDiagnostics,
  configureDiagnostics,
  createProjectLogger,
  getProjectDiagnostics,
  recordDiagnosticTiming,
} from './logger';

beforeEach(() => {
  clearProjectDiagnostics('project-a');
  clearProjectDiagnostics('project-b');
  configureDiagnostics({
    enabled: true,
    level: 'trace',
    namespaces: ['workflows', 'system', 'queue'],
    performanceTimingsEnabled: true,
  });
  vi.setSystemTime(new Date('2026-06-29T00:00:00.000Z'));
});

describe('diagnostics logger', () => {
  it('records entries scoped to the source project', () => {
    const logger = createProjectLogger('workflows', {
      instanceId: 'workflow:center',
      kind: 'widget',
      projectId: 'project-a',
      region: 'center',
      typeId: 'workflow',
    });

    logger.info({ nodes: 120 }, 'Workflow loaded');

    expect(getProjectDiagnostics('project-a')).toMatchObject([
      {
        context: { nodes: 120 },
        createdAt: '2026-06-29T00:00:00.000Z',
        level: 'info',
        message: 'Workflow loaded',
        namespace: 'workflows',
        source: {
          instanceId: 'workflow:center',
          kind: 'widget',
          projectId: 'project-a',
          region: 'center',
          typeId: 'workflow',
        },
      },
    ]);
    expect(getProjectDiagnostics('project-b')).toEqual([]);
  });

  it('records timing entries as normal diagnostics', () => {
    recordDiagnosticTiming({ area: 'persistence', kind: 'workbench', projectId: 'project-a' }, 'workbench:save', 12.34);

    expect(getProjectDiagnostics('project-a')).toMatchObject([
      {
        durationMs: 12.34,
        level: 'debug',
        message: 'workbench:save completed in 12.3ms',
        namespace: 'performance',
        source: { area: 'persistence', kind: 'workbench', projectId: 'project-a' },
      },
    ]);
  });

  it('clears a single project without clearing other projects', () => {
    createProjectLogger('system', { area: 'runtime', kind: 'workbench', projectId: 'project-a' }).error('A failed');
    createProjectLogger('system', { area: 'runtime', kind: 'workbench', projectId: 'project-b' }).warn('B warned');

    clearProjectDiagnostics('project-a');

    expect(getProjectDiagnostics('project-a')).toEqual([]);
    expect(getProjectDiagnostics('project-b')).toHaveLength(1);
  });

  it('filters entries by project logging settings', () => {
    configureDiagnostics({
      enabled: true,
      level: 'warn',
      namespaces: ['system'],
      performanceTimingsEnabled: true,
    });

    createProjectLogger('workflows', { area: 'runtime', kind: 'workbench', projectId: 'project-a' }).error(
      'Wrong namespace'
    );
    createProjectLogger('system', { area: 'runtime', kind: 'workbench', projectId: 'project-a' }).debug('Too quiet');
    createProjectLogger('system', { area: 'runtime', kind: 'workbench', projectId: 'project-a' }).warn('Recorded');

    expect(getProjectDiagnostics('project-a')).toMatchObject([{ level: 'warn', message: 'Recorded' }]);
  });

  it('can disable project diagnostics and performance timings independently', () => {
    configureDiagnostics({
      enabled: false,
      level: 'trace',
      namespaces: ['system'],
      performanceTimingsEnabled: true,
    });

    createProjectLogger('system', { area: 'runtime', kind: 'workbench', projectId: 'project-a' }).error('Skipped');

    configureDiagnostics({
      enabled: true,
      level: 'trace',
      namespaces: ['system'],
      performanceTimingsEnabled: false,
    });

    recordDiagnosticTiming({ area: 'runtime', kind: 'workbench', projectId: 'project-a' }, 'workbench:test', 1.2);

    expect(getProjectDiagnostics('project-a')).toEqual([]);
  });

  it('applies the current diagnostics config to every project', () => {
    configureDiagnostics({
      enabled: true,
      level: 'warn',
      namespaces: ['queue'],
      performanceTimingsEnabled: true,
    });

    createProjectLogger('queue', { area: 'autosave', kind: 'workbench', projectId: 'project-b' }).error('Recorded');

    expect(getProjectDiagnostics('project-b')).toMatchObject([{ level: 'error', message: 'Recorded' }]);
  });
});
