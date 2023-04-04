import { HEADERS } from '../api/core/request';

/**
 * Returns the headers of a given response object
 */
export const getHeaders = (response: any): Record<string, string> => {
  if (!(HEADERS in response)) {
    throw new Error('Response does not have headers');
  }

  return response[HEADERS];
};
