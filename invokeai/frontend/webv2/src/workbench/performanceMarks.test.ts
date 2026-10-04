import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { markWorkbenchPerf, measureWorkbenchPerf, timeWorkbenchPerf } from './performanceMarks';

const source = { area: 'editor', namespace: 'workflows' as const, projectId: 'project-a' };

beforeEach(() => {
  resetLogging();
  configureLogging({ ...DEFAULT_LOGGING_CONFIG, namespaces: [], performanceTimingsEnabled: true });
  vi.spyOn(performance, 'mark').mockImplementation((name: string) => ({ name }) as PerformanceMark);
  vi.spyOn(performance, 'measure').mockImplementation(
    (name: string) => ({ duration: 12.34, name }) as PerformanceMeasure
  );
  vi.clearAllMocks();
});

describe('performanceMarks', () => {
  it('does nothing when no source is provided', () => {
    markWorkbenchPerf('workflow:test');
    measureWorkbenchPerf('workflow:measure', 'workflow:test');

    expect(performance.mark).not.toHaveBeenCalled();
    expect(performance.measure).not.toHaveBeenCalled();
    expect(getLogSnapshot().entries).toEqual([]);
  });

  it('marks, measures and records timings for the provided source', () => {
    const result = timeWorkbenchPerf('workflow:build', source, () => 42);

    expect(result).toBe(42);
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:start');
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:end');
    expect(performance.measure).toHaveBeenCalledWith('workflow:build', 'workflow:build:start', 'workflow:build:end');
    expect(getLogSnapshot().entries).toMatchObject([
      {
        durationMs: 12.34,
        message: 'workflow:build completed in 12.3ms',
        name: 'workflow:build',
        namespace: 'performance',
        source,
      },
    ]);
  });

  it('does not call the Performance API when timing collection is disabled', () => {
    configureLogging({ ...DEFAULT_LOGGING_CONFIG, performanceTimingsEnabled: false });

    timeWorkbenchPerf('workflow:build', source, () => 42);

    expect(performance.mark).not.toHaveBeenCalled();
    expect(performance.measure).not.toHaveBeenCalled();
    expect(getLogSnapshot().entries).toEqual([]);
  });
});
