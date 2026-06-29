import { canRecordDiagnosticTiming, recordDiagnosticTiming, type DiagnosticSource } from './diagnostics/logger';

const hasPerformanceApi = (): boolean => typeof performance !== 'undefined' && typeof performance.mark === 'function';

const canMeasurePerf = (source?: DiagnosticSource): source is DiagnosticSource =>
  hasPerformanceApi() && canRecordDiagnosticTiming(source);

export const markWorkbenchPerf = (name: string, source?: DiagnosticSource): void => {
  if (!canMeasurePerf(source)) {
    return;
  }

  performance.mark(name);
};

export const measureWorkbenchPerf = (
  name: string,
  startMark: string,
  source?: DiagnosticSource,
  endMark?: string
): void => {
  if (!canMeasurePerf(source)) {
    return;
  }

  try {
    const measure = endMark ? performance.measure(name, startMark, endMark) : performance.measure(name, startMark);

    recordDiagnosticTiming(source, measure.name, measure.duration);

    performance.clearMeasures?.(name);
    performance.clearMarks?.(startMark);
    if (endMark) {
      performance.clearMarks?.(endMark);
    }
  } catch {
    // A missing mark should never affect workflow behavior.
  }
};

export const timeWorkbenchPerf = <T>(name: string, source: DiagnosticSource | undefined, callback: () => T): T => {
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
