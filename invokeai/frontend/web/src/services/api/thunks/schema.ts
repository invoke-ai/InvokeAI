import SwaggerParser from '@apidevtools/swagger-parser';
import { createAsyncThunk } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import { OpenAPIV3 } from 'openapi-types';

const schemaLog = log.child({ namespace: 'schema' });

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
    try {
      const dereferencedSchema = (await SwaggerParser.dereference(
        'openapi.json'
      )) as OpenAPIV3.Document;

      const schemaJSON = JSON.parse(
        JSON.stringify(dereferencedSchema, getCircularReplacer())
      );

      return schemaJSON;
    } catch (error) {
      return rejectWithValue({ error });
    }
  }
);
