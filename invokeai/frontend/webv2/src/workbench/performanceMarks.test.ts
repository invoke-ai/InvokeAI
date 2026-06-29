import { beforeEach, describe, expect, it, vi } from 'vitest';

import { clearProjectDiagnostics, configureDiagnostics, getProjectDiagnostics } from './diagnostics/logger';
import { markWorkbenchPerf, measureWorkbenchPerf, timeWorkbenchPerf } from './performanceMarks';

beforeEach(() => {
  clearProjectDiagnostics('project-a');
  configureDiagnostics({
    enabled: true,
    level: 'trace',
    namespaces: [],
    performanceTimingsEnabled: true,
  });
  vi.spyOn(performance, 'mark').mockImplementation((name: string) => ({ name }) as PerformanceMark);
  vi.spyOn(performance, 'measure').mockImplementation(
    (name: string) => ({ duration: 12.34, name }) as PerformanceMeasure
  );
  vi.clearAllMocks();
});

describe('performanceMarks', () => {
  it('does nothing when no diagnostic source is provided', () => {
    markWorkbenchPerf('workflow:test');
    measureWorkbenchPerf('workflow:measure', 'workflow:test');

    expect(performance.mark).not.toHaveBeenCalled();
    expect(performance.measure).not.toHaveBeenCalled();
    expect(getProjectDiagnostics('project-a')).toEqual([]);
  });

  it('marks, measures and records timings for the provided diagnostic source', () => {
    const source = { area: 'test', kind: 'workbench' as const, projectId: 'project-a' };
    const result = timeWorkbenchPerf('workflow:build', source, () => 42);

    expect(result).toBe(42);
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:start');
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:end');
    expect(performance.measure).toHaveBeenCalledWith('workflow:build', 'workflow:build:start', 'workflow:build:end');
    expect(getProjectDiagnostics('project-a')).toMatchObject([
      { durationMs: 12.34, message: 'workflow:build completed in 12.3ms', namespace: 'performance', source },
    ]);
  });

  it('does not call the Performance API when timing collection is disabled', () => {
    configureDiagnostics({
      enabled: true,
      level: 'trace',
      namespaces: [],
      performanceTimingsEnabled: false,
    });

    const source = { area: 'test', kind: 'workbench' as const, projectId: 'project-a' };

    timeWorkbenchPerf('workflow:build', source, () => 42);

    expect(performance.mark).not.toHaveBeenCalled();
    expect(performance.measure).not.toHaveBeenCalled();
    expect(getProjectDiagnostics('project-a')).toEqual([]);
  });
});
