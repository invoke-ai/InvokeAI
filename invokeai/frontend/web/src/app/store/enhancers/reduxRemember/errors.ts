import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { PersistError, RehydrateError } from 'redux-remember';
import { serializeError } from 'serialize-error';

type StorageErrorArgs = {
  key: string;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */ // any is correct
  value?: any;
  originalError?: unknown;
  projectId?: string;
};

export class StorageError extends Error {
  key: string;
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */ // any is correct
  value?: any;
  originalError?: Error;
  projectId?: string;

  constructor({ key, value, originalError, projectId }: StorageErrorArgs) {
    super(`Error setting ${key}`);
    this.name = 'StorageSetError';
    this.key = key;
    if (value !== undefined) {
      this.value = value;
    }
    if (projectId !== undefined) {
      this.projectId = projectId;
    }
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
