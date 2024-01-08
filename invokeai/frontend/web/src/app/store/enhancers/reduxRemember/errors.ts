import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { PersistError, RehydrateError } from 'redux-remember';
import { serializeError } from 'serialize-error';

export class StorageSetError extends Error {
  key: string;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */ // any is correct
  value: any;
  originalError?: Error;

  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */ // any is correct
  constructor(key: string, value: any, originalError?: unknown) {
    super(`Error setting ${key}`);
    this.name = 'StorageSetError';
    this.key = key;
    this.value = value;
    if (originalError instanceof Error) {
      this.originalError = originalError;
    }
  }
}

export class StorageGetError extends Error {
  key: string;
  originalError?: Error;

  constructor(key: string, originalError?: unknown) {
    super(`Error getting ${key}`);
    this.name = 'StorageSetError';
    this.key = key;
    if (originalError instanceof Error) {
      this.originalError = originalError;
    }
  }
}

export const errorHandler = (err: PersistError | RehydrateError) => {
  const log = logger('system');
  if (err instanceof PersistError) {
    log.error({ error: serializeError(err) }, 'Problem persisting state');
  } else if (err instanceof RehydrateError) {
    log.error({ error: serializeError(err) }, 'Problem rehydrating state');
  } else {
    log.error({ error: parseify(err) }, 'Problem in persistence layer');
  }
};
