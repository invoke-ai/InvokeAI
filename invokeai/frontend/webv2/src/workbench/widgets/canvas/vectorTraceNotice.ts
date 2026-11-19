import type { VectorTraceResult } from '@workbench/canvas-engine/api';

/** Maps vector rendering outcomes to the existing Properties and path-menu notices. */
export const vectorTraceErrorKey = ({ status }: VectorTraceResult): string | null => {
  if (status === 'committed' || status === 'aborted' || status === 'stale') {
    return null;
  }
  if (status === 'empty') {
    return 'traceEmpty';
  }
  if (status === 'too-complex') {
    return 'traceTooComplex';
  }
  if (status === 'over-budget') {
    return 'traceOverBudget';
  }
  if (status === 'busy' || status === 'gesture-active') {
    return 'traceBusy';
  }
  return 'traceFailed';
};
