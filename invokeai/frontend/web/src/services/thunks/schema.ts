import { createAsyncThunk } from '@reduxjs/toolkit';
import { parsedOpenAPISchema } from 'features/nodes/store/nodesSlice';
import { OpenAPIV3 } from 'openapi-types';

export const receivedOpenAPISchema = createAsyncThunk(
  'nodes/receivedOpenAPISchema',
  async (_, { dispatch }): Promise<OpenAPIV3.Document> => {
    const response = await fetch(`openapi.json`);
    const openAPISchema = (await response.json()) as OpenAPIV3.Document;

    console.debug('OpenAPI schema: ', openAPISchema);

    dispatch(parsedOpenAPISchema(openAPISchema));

    return openAPISchema;
  }
);
