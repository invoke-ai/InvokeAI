/**
 * Serialize an object to JSON and back to a new object
 */
export const parseify = (obj: unknown) => {
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch {
    return 'Error parsing object';
  }
};
