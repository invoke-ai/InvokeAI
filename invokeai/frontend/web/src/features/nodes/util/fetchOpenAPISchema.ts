import { OpenAPIV3 } from 'openapi-types';

// grab the openapi schema json
export async function fetchOpenAPISchema(): Promise<OpenAPIV3.Document> {
  const response = await fetch(`openapi.json`);
  const jsonData = await response.json();
  return jsonData;
}
