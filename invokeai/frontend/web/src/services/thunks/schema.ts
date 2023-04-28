import { createAsyncThunk } from '@reduxjs/toolkit';
import { parseSchema } from 'features/nodes/util/parseSchema';
import { OpenAPIV3 } from 'openapi-types';

export const receivedOpenAPISchema = createAsyncThunk(
  'nodes/receivedOpenAPISchema',
  async () => {
    const response = await fetch(`openapi.json`);
    const jsonData = (await response.json()) as OpenAPIV3.Document;

    console.debug('OpenAPI schema: ', jsonData);

    const parsedSchema = parseSchema(jsonData);

    console.debug('Parsed schema: ', parsedSchema);

    return parsedSchema;
  }
);
