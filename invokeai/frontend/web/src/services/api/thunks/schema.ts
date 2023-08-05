import { createAsyncThunk } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';

function getCircularReplacer() {
  const ancestors: Record<string, any>[] = [];
  return function (key: string, value: any) {
    if (typeof value !== 'object' || value === null) {
      return value;
    }
    // `this` is the object that value is contained in,
    // i.e., its direct parent.
    // @ts-ignore
    while (ancestors.length > 0 && ancestors.at(-1) !== this) {
      ancestors.pop();
    }
    if (ancestors.includes(value)) {
      return '[Circular]';
    }
    ancestors.push(value);
    return value;
  };
}

export const receivedOpenAPISchema = createAsyncThunk(
  'nodes/receivedOpenAPISchema',
  async (_, { dispatch, rejectWithValue }) => {
    const log = logger('system');
    try {
      const response = await fetch(`openapi.json`);
      const openAPISchema = await response.json();

      log.info({ openAPISchema }, 'Received OpenAPI schema');

      const schemaJSON = JSON.parse(
        JSON.stringify(openAPISchema, getCircularReplacer())
      );

      return schemaJSON;
    } catch (error) {
      return rejectWithValue({ error });
    }
  }
);
