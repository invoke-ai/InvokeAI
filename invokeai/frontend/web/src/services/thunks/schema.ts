import { createAsyncThunk } from '@reduxjs/toolkit';
import { OpenAPIV3 } from 'openapi-types';

export const receivedOpenAPISchema = createAsyncThunk(
  'nodes/receivedOpenAPISchema',
  async () => {
    const response = await fetch(`openapi.json`);
    const jsonData = (await response.json()) as OpenAPIV3.Document;

    console.debug('OpenAPI schema: ', jsonData);

    return jsonData;
  }
);
