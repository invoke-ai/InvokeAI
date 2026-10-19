export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export const LOG_LEVELS: readonly LogLevel[] = ['trace', 'debug', 'info', 'warn', 'error', 'fatal'];

export const LOG_LEVEL_ORDER: Record<LogLevel, number> = {
  debug: 20,
  error: 50,
  fatal: 60,
  info: 30,
  trace: 10,
  warn: 40,
};

/** Recording namespaces users can select; `performance` timings are toggled separately. */
export const LOG_NAMESPACES = [
  'app',
  'canvas',
  'canvas-workflow-integration',
  'config',
  'dnd',
  'events',
  'gallery',
  'generation',
  'metadata',
  'models',
  'persistence',
  'queue',
  'system',
  'transport',
  'workflows',
] as const;

export type LogNamespace = (typeof LOG_NAMESPACES)[number];

export const isLogLevel = (value: unknown): value is LogLevel => LOG_LEVELS.includes(value as LogLevel);

export const isLogNamespace = (value: unknown): value is LogNamespace => LOG_NAMESPACES.includes(value as LogNamespace);

export interface LogWidgetSource {
  instanceId: string;
  region: string;
  typeId: string;
}

/** Who recorded an event. Namespace and area are required; the rest attributes work to a project or operation. */
export interface LogSource {
  area: string;
  namespace: LogNamespace;
  operationId?: string;
  projectId?: string;
  widget?: LogWidgetSource;
}

export type LogContext = Record<string, unknown>;

export interface LogEventInput {
  /** Stable machine name such as `queue.submission-failed`; defaults to the source area. */
  name?: string;
  message: string;
  error?: unknown;
  /** A function defers construction until the event passes recording filters. */
  context?: LogContext | (() => LogContext);
}

export type LogInput = string | LogEventInput;

export interface Logger {
  trace(input: LogInput): void;
  debug(input: LogInput): void;
  info(input: LogInput): void;
  warn(input: LogInput): void;
  error(input: LogInput): void;
  fatal(input: LogInput): void;
  /** A logger recording under this one's source merged with `source`. */
  child(source: Partial<LogSource>): Logger;
}

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface SerializedError {
  cause?: SerializedError | JsonValue;
  code?: string | number;
  message: string;
  name: string;
  stack?: string;
  status?: number;
}

export interface LogEntry {
  readonly context?: JsonValue;
  readonly createdAt: string;
  readonly durationMs?: number;
  readonly error?: SerializedError;
  readonly id: string;
  readonly level: LogLevel;
  readonly message: string;
  readonly name: string;
  readonly namespace: LogNamespace | 'performance';
  readonly sequence: number;
  readonly source: LogSource;
  /** Present when normalization dropped or shortened part of the payload. */
  readonly truncated?: true;
}

export interface LoggingConfig {
  consoleOutputEnabled: boolean;
  enabled: boolean;
  level: LogLevel;
  namespaces: readonly LogNamespace[];
  performanceTimingsEnabled: boolean;
}

export const DEFAULT_LOGGING_CONFIG: LoggingConfig = Object.freeze({
  consoleOutputEnabled: false,
  enabled: true,
  level: 'warn',
  namespaces: LOG_NAMESPACES,
  performanceTimingsEnabled: false,
});

export interface LogRetention {
  readonly problems: { readonly count: number; readonly evicted: number; readonly limit: number };
  readonly timings: { readonly count: number; readonly evicted: number; readonly limit: number };
  readonly verbose: { readonly count: number; readonly evicted: number; readonly limit: number };
}

export interface LogSnapshot {
  /** All retained entries, newest first. */
  readonly entries: readonly LogEntry[];
  readonly retention: LogRetention;
  /** Entries accepted since the last clear whose payload was truncated. */
  readonly truncatedCount: number;
}

/** Which entries a viewer shows or clears. Application events carry no project id. */
export type LogScope =
  | { kind: 'all' }
  | { kind: 'project'; projectId: string }
  | { kind: 'project-and-application'; projectId: string };

export const matchesLogScope = (entry: LogEntry, scope: LogScope): boolean => {
  switch (scope.kind) {
    case 'all':
      return true;
    case 'project':
      return entry.source.projectId === scope.projectId;
    case 'project-and-application':
      return entry.source.projectId === undefined || entry.source.projectId === scope.projectId;
  }
};
