/* oxlint-disable no-console */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getWorkbenchPerfEntries,
  markWorkbenchPerf,
  measureWorkbenchPerf,
  timeWorkbenchPerf,
} from './performanceMarks';

const storage = new Map<string, string>();

vi.stubGlobal('window', {
  localStorage: {
    getItem: (key: string): string | null => storage.get(key) ?? null,
  },
});

beforeEach(() => {
  storage.clear();
  window.__invokeaiWorkbenchPerf = [];
  vi.spyOn(performance, 'mark').mockImplementation((name: string) => ({ name }) as PerformanceMark);
  vi.spyOn(performance, 'measure').mockImplementation(
    (name: string) => ({ duration: 12.34, name }) as PerformanceMeasure
  );
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
});

describe('performanceMarks', () => {
  it('does nothing when workbench perf logging is not enabled', () => {
    markWorkbenchPerf('workflow:test');
    measureWorkbenchPerf('workflow:measure', 'workflow:test');

    expect(performance.mark).not.toHaveBeenCalled();
    expect(performance.measure).not.toHaveBeenCalled();
    expect(console.info).not.toHaveBeenCalled();
    expect(getWorkbenchPerfEntries()).toEqual([]);
  });

  it('marks and measures when enabled', () => {
    storage.set('invokeai:webv2:perf', '1');

    const result = timeWorkbenchPerf('workflow:build', () => 42);

    expect(result).toBe(42);
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:start');
    expect(performance.mark).toHaveBeenCalledWith('workflow:build:end');
    expect(performance.measure).toHaveBeenCalledWith('workflow:build', 'workflow:build:start', 'workflow:build:end');
    expect(console.info).toHaveBeenCalledWith('[workbench perf] workflow:build: 12.3ms');
    expect(getWorkbenchPerfEntries()).toEqual([
      { duration: 12.34, name: 'workflow:build', timestamp: expect.any(Number) },
    ]);
  });
});
