import { createLogWriter } from '@roarr/browser-log-writer';
import { atom } from 'nanostores';
import type { Logger, MessageSerializer } from 'roarr';
import { ROARR, Roarr } from 'roarr';
import { z } from 'zod';

const serializeMessage: MessageSerializer = (message) => {
  return JSON.stringify(message);
};

ROARR.serializeMessage = serializeMessage;
ROARR.write = createLogWriter();

export const BASE_CONTEXT = {};

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
  | 'queue'
  | 'dnd';

export const logger = (namespace: LoggerNamespace) => $logger.get().child({ namespace });

export const zLogLevel = z.enum(['trace', 'debug', 'info', 'warn', 'error', 'fatal']);
export type LogLevel = z.infer<typeof zLogLevel>;
export const isLogLevel = (v: unknown): v is LogLevel => zLogLevel.safeParse(v).success;

// Translate human-readable log levels to numbers, used for log filtering
export const LOG_LEVEL_MAP: Record<LogLevel, number> = {
  trace: 10,
  debug: 20,
  info: 30,
  warn: 40,
  error: 50,
  fatal: 60,
};
