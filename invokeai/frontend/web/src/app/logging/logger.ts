import { createLogWriter } from '@roarr/browser-log-writer';
import { atom } from 'nanostores';
import { Logger, ROARR, Roarr } from 'roarr';

ROARR.write = createLogWriter();

export const BASE_CONTEXT = {};
export const log = Roarr.child(BASE_CONTEXT);

export const $logger = atom<Logger>(Roarr.child(BASE_CONTEXT));

export type LoggerNamespace =
  | 'images'
  | 'models'
  | 'config'
  | 'canvas'
  | 'txt2img'
  | 'img2img'
  | 'nodes'
  | 'system'
  | 'socketio'
  | 'session'
  | 'dnd';

export const logger = (namespace: LoggerNamespace) =>
  $logger.get().child({ namespace });

export const VALID_LOG_LEVELS = [
  'trace',
  'debug',
  'info',
  'warn',
  'error',
  'fatal',
] as const;

export type InvokeLogLevel = (typeof VALID_LOG_LEVELS)[number];

// Translate human-readable log levels to numbers, used for log filtering
export const LOG_LEVEL_MAP: Record<InvokeLogLevel, number> = {
  trace: 10,
  debug: 20,
  info: 30,
  warn: 40,
  error: 50,
  fatal: 60,
};
