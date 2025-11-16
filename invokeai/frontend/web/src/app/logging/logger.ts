import { createLogWriter } from '@roarr/browser-log-writer';
import { atom } from 'nanostores';
import type { Logger, MessageSerializer } from 'roarr';
import { ROARR, Roarr } from 'roarr';
import { z } from 'zod';

const serializeMessage: MessageSerializer = (message) => {
  return JSON.stringify(message);
};

ROARR.serializeMessage = serializeMessage;

const BASE_CONTEXT = {};

const $logger = atom<Logger>(Roarr.child(BASE_CONTEXT));

export const zLogNamespace = z.enum([
  'canvas',
  'config',
  'dnd',
  'events',
  'gallery',
  'generation',
  'metadata',
  'models',
  'system',
  'queue',
  'workflows',
]);
export type LogNamespace = z.infer<typeof zLogNamespace>;

export const logger = (namespace: LogNamespace) => $logger.get().child({ namespace });

export const zLogLevel = z.enum(['trace', 'debug', 'info', 'warn', 'error', 'fatal']);
export type LogLevel = z.infer<typeof zLogLevel>;
export const isLogLevel = (v: unknown): v is LogLevel => zLogLevel.safeParse(v).success;

// Translate human-readable log levels to numbers, used for log filtering
const LOG_LEVEL_MAP: Record<LogLevel, number> = {
  trace: 10,
  debug: 20,
  info: 30,
  warn: 40,
  error: 50,
  fatal: 60,
};

/**
 * Configure logging, pushing settings to local storage.
 *
 * @param logIsEnabled Whether logging is enabled
 * @param logLevel The log level
 * @param logNamespaces A list of log namespaces to enable, or '*' to enable all
 */
export const configureLogging = (
  logIsEnabled: boolean = true,
  logLevel: LogLevel = 'warn',
  logNamespaces: LogNamespace[] | '*'
): void => {
  if (!logIsEnabled) {
    // Disable console log output
    localStorage.setItem('ROARR_LOG', 'false');
  } else {
    // Enable console log output
    localStorage.setItem('ROARR_LOG', 'true');

    // Use a filter to show only logs of the given level
    let filter = `context.logLevel:>=${LOG_LEVEL_MAP[logLevel]}`;

    const namespaces = logNamespaces === '*' ? zLogNamespace.options : logNamespaces;

    if (namespaces.length > 0) {
      filter += ` AND (${namespaces.map((ns) => `context.namespace:${ns}`).join(' OR ')})`;
    } else {
      // This effectively hides all logs because we use namespaces for all logs
      filter += ' AND context.namespace:undefined';
    }

    localStorage.setItem('ROARR_FILTER', filter);
  }

  const styleOutput = localStorage.getItem('ROARR_STYLE_OUTPUT') === 'false' ? false : true;

  ROARR.write = createLogWriter({ styleOutput });
};
