/* oxlint-disable no-console */

const PERF_STORAGE_KEY = 'invokeai:webv2:perf';
const MAX_ENTRIES = 200;

interface WorkbenchPerfEntry {
  duration: number;
  name: string;
  timestamp: number;
}

declare global {
  interface Window {
    __invokeaiWorkbenchPerf?: WorkbenchPerfEntry[];
  }
}

const hasPerformanceApi = (): boolean => typeof performance !== 'undefined' && typeof performance.mark === 'function';

const isPerfEnabled = (): boolean => {
  if (!hasPerformanceApi()) {
    return false;
  }

  try {
    return typeof window !== 'undefined' && window.localStorage.getItem(PERF_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
};

export const markWorkbenchPerf = (name: string): void => {
  if (!isPerfEnabled()) {
    return;
  }

  performance.mark(name);
};

const recordWorkbenchPerfEntry = (entry: WorkbenchPerfEntry): void => {
  if (typeof window === 'undefined') {
    return;
  }

  // TODO: Move benchmark collection out of a global window buffer when a real diagnostics surface exists.
  const entries = window.__invokeaiWorkbenchPerf ?? [];

  entries.push(entry);

  if (entries.length > MAX_ENTRIES) {
    entries.splice(0, entries.length - MAX_ENTRIES);
  }

  window.__invokeaiWorkbenchPerf = entries;
};

export const measureWorkbenchPerf = (name: string, startMark: string, endMark?: string): void => {
  if (!isPerfEnabled()) {
    return;
  }

  try {
    const measure = endMark ? performance.measure(name, startMark, endMark) : performance.measure(name, startMark);
    const entry = { duration: measure.duration, name: measure.name, timestamp: performance.now() };

    recordWorkbenchPerfEntry(entry);

    // TODO: Replace console logging with a dedicated benchmark sink/export UI before keeping this long-term.
    console.info(`[workbench perf] ${entry.name}: ${entry.duration.toFixed(1)}ms`);
  } catch {
    // A missing mark should never affect workflow behavior.
  }
};

export const getWorkbenchPerfEntries = (): WorkbenchPerfEntry[] => {
  if (typeof window === 'undefined') {
    return [];
  }

  return [...(window.__invokeaiWorkbenchPerf ?? [])];
};

export const timeWorkbenchPerf = <T>(name: string, callback: () => T): T => {
  if (!isPerfEnabled()) {
    return callback();
  }

  const startMark = `${name}:start`;
  const endMark = `${name}:end`;

  markWorkbenchPerf(startMark);

  try {
    return callback();
  } finally {
    markWorkbenchPerf(endMark);
    measureWorkbenchPerf(name, startMark, endMark);
  }
};
