import type { LogEntry, LoggingConfig, LogRetention, LogScope } from './contracts';

export const LOG_EXPORT_FORMAT = 'invokeai-webv2-logs';
export const LOG_EXPORT_VERSION = 1;

export interface LogExportEnvelope {
  application: { version: string };
  capture: LoggingConfig;
  entries: readonly LogEntry[];
  entryCount: number;
  exportedAt: string;
  format: typeof LOG_EXPORT_FORMAT;
  retention: LogRetention;
  scope: LogScope;
  version: typeof LOG_EXPORT_VERSION;
}

/** A versioned envelope so support tooling can tell captured settings and retention from the entries. */
export const createLogExport = (
  entries: readonly LogEntry[],
  options: { appVersion: string; capture: LoggingConfig; retention: LogRetention; scope: LogScope }
): LogExportEnvelope => ({
  application: { version: options.appVersion },
  capture: options.capture,
  entries,
  entryCount: entries.length,
  exportedAt: new Date().toISOString(),
  format: LOG_EXPORT_FORMAT,
  retention: options.retention,
  scope: options.scope,
  version: LOG_EXPORT_VERSION,
});
