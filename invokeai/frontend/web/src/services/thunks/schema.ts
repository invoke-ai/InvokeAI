import { createAsyncThunk } from '@reduxjs/toolkit';
import { log } from 'app/logging/useLogger';
import { parsedOpenAPISchema } from 'features/nodes/store/nodesSlice';
import { OpenAPIV3 } from 'openapi-types';

const schemaLog = log.child({ namespace: 'schema' });

export const receivedOpenAPISchema = createAsyncThunk(
  'nodes/receivedOpenAPISchema',
  async (_, { dispatch }): Promise<OpenAPIV3.Document> => {
    const response = await fetch(`openapi.json`);
    const openAPISchema = await response.json();

    schemaLog.info({ openAPISchema }, 'Received OpenAPI schema');

    dispatch(parsedOpenAPISchema(openAPISchema as OpenAPIV3.Document));

    return openAPISchema;
  }
);
