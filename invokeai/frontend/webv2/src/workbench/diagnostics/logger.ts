import type { DeveloperLogLevel, DeveloperLogNamespace } from '@workbench/diagnostics/contracts';
import type { WidgetContributionSource } from '@workbench/widgetContracts';

import { useSyncExternalStore } from 'react';

const MAX_PROJECT_DIAGNOSTIC_ENTRIES = 500;
const EMPTY_DIAGNOSTIC_ENTRIES: DiagnosticEntry[] = [];

const LOG_LEVEL_ORDER: Record<DeveloperLogLevel, number> = {
  debug: 20,
  error: 50,
  fatal: 60,
  info: 30,
  trace: 10,
  warn: 40,
};

export type DiagnosticSource =
  | ({ kind: 'widget' } & WidgetContributionSource)
  | { area: string; kind: 'workbench'; projectId?: string };

export interface DiagnosticEntry {
  context?: Record<string, unknown>;
  createdAt: string;
  durationMs?: number;
  id: string;
  level: DeveloperLogLevel;
  message: string;
  namespace: DeveloperLogNamespace | 'performance';
  source: DiagnosticSource;
}

export interface DiagnosticsConfig {
  enabled: boolean;
  level: DeveloperLogLevel;
  namespaces: DeveloperLogNamespace[];
  performanceTimingsEnabled: boolean;
}

type DiagnosticListener = () => void;

type LogArgument = string | Record<string, unknown>;

const entriesByProjectId = new Map<string, DiagnosticEntry[]>();
const listenersByProjectId = new Map<string, Set<DiagnosticListener>>();

let diagnosticsConfig: DiagnosticsConfig = {
  enabled: true,
  level: 'warn',
  namespaces: ['system', 'queue', 'workflows'],
  performanceTimingsEnabled: false,
};

let nextDiagnosticEntryId = 0;

const getProjectIdFromSource = (source: DiagnosticSource): string | null => source.projectId ?? null;

const shouldRecordEntry = (entry: Omit<DiagnosticEntry, 'createdAt' | 'id'>): boolean => {
  if (entry.namespace === 'performance') {
    return diagnosticsConfig.performanceTimingsEnabled;
  }

  return (
    diagnosticsConfig.enabled &&
    LOG_LEVEL_ORDER[entry.level] >= LOG_LEVEL_ORDER[diagnosticsConfig.level] &&
    diagnosticsConfig.namespaces.includes(entry.namespace)
  );
};

const getProjectEntries = (projectId: string): DiagnosticEntry[] => {
  const entries = entriesByProjectId.get(projectId) ?? [];

  if (!entriesByProjectId.has(projectId)) {
    entriesByProjectId.set(projectId, entries);
  }

  return entries;
};

const getProjectEntriesSnapshot = (projectId: string): DiagnosticEntry[] =>
  entriesByProjectId.get(projectId) ?? EMPTY_DIAGNOSTIC_ENTRIES;

const notifyProjectDiagnostics = (projectId: string): void => {
  for (const listener of listenersByProjectId.get(projectId) ?? []) {
    listener();
  }
};

const normalizeLogArgs = (
  first: LogArgument,
  second?: string
): { context?: Record<string, unknown>; message: string } => {
  if (typeof first === 'string') {
    return { message: first };
  }

  return { context: first, message: second ?? '' };
};

export const recordDiagnosticEntry = (entry: Omit<DiagnosticEntry, 'createdAt' | 'id'>): DiagnosticEntry | null => {
  const projectId = getProjectIdFromSource(entry.source);

  if (!projectId) {
    return null;
  }

  if (!shouldRecordEntry(entry)) {
    return null;
  }

  const nextEntry: DiagnosticEntry = {
    ...entry,
    createdAt: new Date().toISOString(),
    id: `diagnostic-${nextDiagnosticEntryId++}`,
  };
  const entries = [nextEntry, ...getProjectEntries(projectId)].slice(0, MAX_PROJECT_DIAGNOSTIC_ENTRIES);

  entriesByProjectId.set(projectId, entries);

  notifyProjectDiagnostics(projectId);

  return nextEntry;
};

export const createProjectLogger = (namespace: DeveloperLogNamespace, source: DiagnosticSource) => {
  const log = (level: DeveloperLogLevel, first: LogArgument, second?: string): void => {
    const { context, message } = normalizeLogArgs(first, second);

    recordDiagnosticEntry({ context, level, message, namespace, source });
  };

  return {
    debug: (first: LogArgument, second?: string) => log('debug', first, second),
    error: (first: LogArgument, second?: string) => log('error', first, second),
    fatal: (first: LogArgument, second?: string) => log('fatal', first, second),
    info: (first: LogArgument, second?: string) => log('info', first, second),
    trace: (first: LogArgument, second?: string) => log('trace', first, second),
    warn: (first: LogArgument, second?: string) => log('warn', first, second),
  };
};

export const configureDiagnostics = (config: DiagnosticsConfig): void => {
  diagnosticsConfig = { ...config, namespaces: [...config.namespaces] };
};

export const canRecordDiagnosticTiming = (source?: DiagnosticSource): source is DiagnosticSource => {
  if (!source?.projectId) {
    return false;
  }

  return diagnosticsConfig.performanceTimingsEnabled;
};

export const recordDiagnosticTiming = (
  source: DiagnosticSource,
  name: string,
  durationMs: number
): DiagnosticEntry | null =>
  recordDiagnosticEntry({
    durationMs,
    level: 'debug',
    message: `${name} completed in ${durationMs.toFixed(1)}ms`,
    namespace: 'performance',
    source,
  });

export const getProjectDiagnostics = (projectId: string): DiagnosticEntry[] => [...getProjectEntries(projectId)];

export const useProjectDiagnostics = (projectId: string): DiagnosticEntry[] =>
  useSyncExternalStore(
    (listener) => subscribeProjectDiagnostics(projectId, listener),
    () => getProjectEntriesSnapshot(projectId),
    () => EMPTY_DIAGNOSTIC_ENTRIES
  );

export const clearProjectDiagnostics = (projectId: string): void => {
  entriesByProjectId.delete(projectId);
  notifyProjectDiagnostics(projectId);
};

export const subscribeProjectDiagnostics = (projectId: string, listener: DiagnosticListener): (() => void) => {
  const listeners = listenersByProjectId.get(projectId) ?? new Set<DiagnosticListener>();

  listeners.add(listener);
  listenersByProjectId.set(projectId, listeners);

  return () => {
    listeners.delete(listener);

    if (listeners.size === 0) {
      listenersByProjectId.delete(projectId);
    }
  };
};
