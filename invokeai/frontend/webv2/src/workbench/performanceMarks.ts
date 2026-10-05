import type { LogSource } from '@platform/logging/contracts';

import { canRecordTiming, recordTiming } from '@platform/logging/logger';

const hasPerformanceApi = (): boolean => typeof performance !== 'undefined' && typeof performance.mark === 'function';

const canMeasurePerf = (source?: LogSource): source is LogSource =>
  source !== undefined && hasPerformanceApi() && canRecordTiming();

export const markWorkbenchPerf = (name: string, source?: LogSource): void => {
  if (!canMeasurePerf(source)) {
    return;
  }

  performance.mark(name);
};

export const measureWorkbenchPerf = (name: string, startMark: string, source?: LogSource, endMark?: string): void => {
  if (!canMeasurePerf(source)) {
    return;
  }

  try {
    const measure = endMark ? performance.measure(name, startMark, endMark) : performance.measure(name, startMark);

    recordTiming(source, measure.name, measure.duration);

    performance.clearMeasures?.(name);
    performance.clearMarks?.(startMark);
    if (endMark) {
      performance.clearMarks?.(endMark);
    }
  } catch {
    // A missing mark should never affect workflow behavior.
  }
};

export const timeWorkbenchPerf = <T>(name: string, source: LogSource | undefined, callback: () => T): T => {
  if (!canMeasurePerf(source)) {
    return callback();
  }

  const startMark = `${name}:start`;
  const endMark = `${name}:end`;

  markWorkbenchPerf(startMark, source);

  try {
    return callback();
  } finally {
    markWorkbenchPerf(endMark, source);
    measureWorkbenchPerf(name, startMark, source, endMark);
  }
};
