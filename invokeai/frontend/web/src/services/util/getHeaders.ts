import { HEADERS } from '../api/core/request';

/**
 * Returns the response headers of the response received by the generated API client.
 */
export const getHeaders = (response: any): Record<string, string> => {
  if (!(HEADERS in response)) {
    throw new Error('Response does not have headers');
  }

  return response[HEADERS];
};
