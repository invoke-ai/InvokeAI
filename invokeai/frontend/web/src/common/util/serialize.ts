/**
 * Serialize an object to JSON and back to a new object
 */
export const parseify = (obj: unknown) => JSON.parse(JSON.stringify(obj));
