import {
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createListenerChannel } from '@platform/state/externalStoreCore';

import type {
  JsonValue,
  LogEntry,
  LogEventInput,
  Logger,
  LoggingConfig,
  LogInput,
  LogLevel,
  LogNamespace,
  LogRetention,
  LogScope,
  LogSnapshot,
  LogSource,
  SerializedError,
} from './contracts';

import { DEFAULT_LOGGING_CONFIG, LOG_LEVEL_ORDER, matchesLogScope } from './contracts';
import { MAX_ENTRY_BYTES, measureJsonBytes, normalizeContext, normalizeError, normalizeMessage } from './normalize';

export const LOG_RETENTION_LIMITS = Object.freeze({ problems: 500, timings: 250, verbose: 1_000 });

const PROBLEM_LEVELS: ReadonlySet<LogLevel> = new Set(['warn', 'error', 'fatal']);
const CONSOLE_METHODS: Record<LogLevel, 'debug' | 'error' | 'info' | 'warn'> = {
  debug: 'debug',
  error: 'error',
  fatal: 'error',
  info: 'info',
  trace: 'debug',
  warn: 'warn',
};

interface BoundedBuffer {
  /** Oldest first. */
  entries: LogEntry[];
  evicted: number;
  readonly limit: number;
}

const createBuffer = (limit: number): BoundedBuffer => ({ entries: [], evicted: 0, limit });

const pushBounded = (buffer: BoundedBuffer, entry: LogEntry): void => {
  buffer.entries.push(entry);

  if (buffer.entries.length > buffer.limit) {
    buffer.entries.splice(0, buffer.entries.length - buffer.limit);
    buffer.evicted += 1;
  }
};

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) {
      deepFreeze(child);
    }
  }

  return value;
};

const cloneSource = (source: LogSource): LogSource => {
  const copy: LogSource = { area: source.area, namespace: source.namespace };

  if (source.projectId !== undefined) {
    copy.projectId = source.projectId;
  }
  if (source.operationId !== undefined) {
    copy.operationId = source.operationId;
  }
  if (source.widget) {
    copy.widget = { instanceId: source.widget.instanceId, region: source.widget.region, typeId: source.widget.typeId };
  }

  return copy;
};

const channel = createListenerChannel();
let config: LoggingConfig = DEFAULT_LOGGING_CONFIG;
let problems = createBuffer(LOG_RETENTION_LIMITS.problems);
let verbose = createBuffer(LOG_RETENTION_LIMITS.verbose);
let timings = createBuffer(LOG_RETENTION_LIMITS.timings);
let truncatedCount = 0;
let nextSequence = 1;
let snapshot: LogSnapshot | null = null;
let publishEpoch = 0;
let isPublishScheduled = false;

const buildRetention = (): LogRetention => ({
  problems: { count: problems.entries.length, evicted: problems.evicted, limit: problems.limit },
  timings: { count: timings.entries.length, evicted: timings.evicted, limit: timings.limit },
  verbose: { count: verbose.entries.length, evicted: verbose.evicted, limit: verbose.limit },
});

const invalidateSnapshot = (): void => {
  snapshot = null;
};

const notifyListeners = (): void => {
  try {
    channel.notify();
  } catch {
    // A failing subscriber must not break the action that produced the event.
  }
};

/** Coalesce publication once per microtask; clears and account transitions publish synchronously instead. */
const schedulePublish = (): void => {
  if (isPublishScheduled) {
    return;
  }

  isPublishScheduled = true;
  const epoch = publishEpoch;

  queueMicrotask(() => {
    if (epoch !== publishEpoch) {
      return;
    }

    isPublishScheduled = false;
    notifyListeners();
  });
};

const publishNow = (): void => {
  publishEpoch += 1;
  isPublishScheduled = false;
  notifyListeners();
};

const shouldRecord = (level: LogLevel, namespace: LogNamespace): boolean =>
  config.enabled && LOG_LEVEL_ORDER[level] >= LOG_LEVEL_ORDER[config.level] && config.namespaces.includes(namespace);

const resolveContext = (context: LogEventInput['context']): unknown => {
  if (typeof context !== 'function') {
    return context;
  }

  try {
    return context();
  } catch (error) {
    return { '[context error]': error instanceof Error ? error.message : String(error) };
  }
};

interface MutableEntry {
  context?: JsonValue;
  createdAt: string;
  durationMs?: number;
  error?: SerializedError;
  id: string;
  level: LogLevel;
  message: string;
  name: string;
  namespace: LogNamespace | 'performance';
  sequence: number;
  source: LogSource;
  truncated?: true;
}

const enforceEntryCeiling = (entry: MutableEntry): void => {
  let bytes = measureJsonBytes(entry);

  if (bytes <= MAX_ENTRY_BYTES) {
    return;
  }

  entry.truncated = true;

  if (entry.context !== undefined) {
    entry.context = `[context omitted: ${measureJsonBytes(entry.context)} bytes]`;
    bytes = measureJsonBytes(entry);
  }

  if (bytes > MAX_ENTRY_BYTES && entry.error?.stack) {
    entry.error = { ...entry.error, stack: `${entry.error.stack.slice(0, 1_024)}…` };
    bytes = measureJsonBytes(entry);
  }

  if (bytes > MAX_ENTRY_BYTES && entry.error?.cause !== undefined) {
    entry.error = { ...entry.error, cause: '[cause omitted]' };
    bytes = measureJsonBytes(entry);
  }

  if (bytes > MAX_ENTRY_BYTES) {
    entry.message = `${entry.message.slice(0, 512)}…`;
  }
};

const createEntry = (
  level: LogLevel,
  namespace: LogNamespace | 'performance',
  source: LogSource,
  input: LogInput,
  durationMs?: number
): LogEntry => {
  const event: LogEventInput = typeof input === 'string' ? { message: input } : input;
  const sequence = nextSequence;

  nextSequence += 1;

  const message = normalizeMessage(typeof event.message === 'string' ? event.message : String(event.message));
  const entry: MutableEntry = {
    createdAt: new Date().toISOString(),
    id: `log-${sequence}`,
    level,
    message: message.value,
    name: event.name ?? source.area,
    namespace,
    sequence,
    source: cloneSource(source),
  };
  let truncated = message.truncated;

  if (durationMs !== undefined) {
    entry.durationMs = durationMs;
  }

  if (event.context !== undefined) {
    const normalized = normalizeContext(resolveContext(event.context));

    entry.context = normalized.value;
    truncated ||= normalized.truncated;
  }

  if (event.error !== undefined) {
    const normalized = normalizeError(event.error);

    entry.error = normalized.error;
    truncated ||= normalized.truncated;
  }

  if (truncated) {
    entry.truncated = true;
  }

  enforceEntryCeiling(entry);

  return deepFreeze(entry) as LogEntry;
};

const writeConsole = (entry: LogEntry): void => {
  if (!config.consoleOutputEnabled) {
    return;
  }

  try {
    const label = `[invoke:${entry.namespace}/${entry.source.area}] ${entry.message}`;
    const payload: Record<string, unknown> = {};

    if (entry.context !== undefined) {
      payload.context = entry.context;
    }
    if (entry.error) {
      payload.error = entry.error;
    }
    if (entry.source.projectId) {
      payload.projectId = entry.source.projectId;
    }
    if (entry.durationMs !== undefined) {
      payload.durationMs = entry.durationMs;
    }

    // oxlint-disable-next-line no-console -- the console sink is the opt-in output destination
    console[CONSOLE_METHODS[entry.level]](label, payload);
  } catch {
    // Console failures never affect the originating action.
  }
};

const store = (entry: LogEntry): void => {
  if (entry.namespace === 'performance') {
    pushBounded(timings, entry);
  } else if (PROBLEM_LEVELS.has(entry.level)) {
    pushBounded(problems, entry);
  } else {
    pushBounded(verbose, entry);
  }

  if (entry.truncated) {
    truncatedCount += 1;
  }

  invalidateSnapshot();
  writeConsole(entry);
  schedulePublish();
};

/** Record one event when recording filters accept it. Failures inside recording never reach the caller. */
export const recordLogEvent = (level: LogLevel, source: LogSource, input: LogInput, owner?: AccountScope): void => {
  if (owner && !isAccountScopeCurrent(owner)) {
    return;
  }

  if (!shouldRecord(level, source.namespace)) {
    return;
  }

  try {
    store(createEntry(level, source.namespace, source, input));
  } catch {
    // Normalization is defensive; a failure here must still be isolated from the caller.
  }
};

export const canRecordTiming = (): boolean => config.performanceTimingsEnabled;

export const recordTiming = (source: LogSource, name: string, durationMs: number, owner?: AccountScope): void => {
  if (!config.performanceTimingsEnabled || (owner && !isAccountScopeCurrent(owner))) {
    return;
  }

  try {
    store(
      createEntry(
        'debug',
        'performance',
        source,
        { message: `${name} completed in ${durationMs.toFixed(1)}ms`, name },
        durationMs
      )
    );
  } catch {
    // Timings are best-effort.
  }
};

const mergeSource = (base: LogSource, override: Partial<LogSource>): LogSource => {
  const merged: LogSource = { ...base, ...override };

  return cloneSource(merged);
};

/**
 * A logger bound to `source`. Pass `owner` from a runtime or operation start so writes after that account
 * lifetime ends are discarded; module-level loggers stay unbound.
 */
export const createLogger = (source: LogSource, options: { owner?: AccountScope } = {}): Logger => {
  const bound = cloneSource(source);
  const write = (level: LogLevel, input: LogInput): void => recordLogEvent(level, bound, input, options.owner);

  return {
    child: (override) => createLogger(mergeSource(bound, override), options),
    debug: (input) => write('debug', input),
    error: (input) => write('error', input),
    fatal: (input) => write('fatal', input),
    info: (input) => write('info', input),
    trace: (input) => write('trace', input),
    warn: (input) => write('warn', input),
  };
};

export const configureLogging = (next: LoggingConfig): void => {
  config = Object.freeze({
    consoleOutputEnabled: next.consoleOutputEnabled,
    enabled: next.enabled,
    level: next.level,
    namespaces: Object.freeze([...next.namespaces]),
    performanceTimingsEnabled: next.performanceTimingsEnabled,
  });
  publishNow();
};

export const getLoggingConfig = (): LoggingConfig => config;

export const getLogSnapshot = (): LogSnapshot => {
  if (!snapshot) {
    const entries = [...problems.entries, ...verbose.entries, ...timings.entries].sort(
      (left, right) => right.sequence - left.sequence
    );

    snapshot = Object.freeze({ entries: Object.freeze(entries), retention: buildRetention(), truncatedCount });
  }

  return snapshot;
};

export const subscribeLogs = (listener: () => void): (() => void) => channel.subscribe(listener);

const removeMatching = (buffer: BoundedBuffer, scope: LogScope): void => {
  buffer.entries = buffer.entries.filter((entry) => !matchesLogScope(entry, scope));
};

/** Remove retained entries in `scope`; clearing everything also resets eviction and truncation counters. */
export const clearLogs = (scope: LogScope = { kind: 'all' }): void => {
  if (scope.kind === 'all') {
    problems = createBuffer(LOG_RETENTION_LIMITS.problems);
    verbose = createBuffer(LOG_RETENTION_LIMITS.verbose);
    timings = createBuffer(LOG_RETENTION_LIMITS.timings);
    truncatedCount = 0;
  } else {
    removeMatching(problems, scope);
    removeMatching(verbose, scope);
    removeMatching(timings, scope);
  }

  invalidateSnapshot();
  publishNow();
};

/** Drop history and pending publications and restore default capture settings. */
export const resetLogging = (): void => {
  config = DEFAULT_LOGGING_CONFIG;
  clearLogs();
};

registerAccountOwnedResource({ clear: resetLogging, name: 'logging' });
