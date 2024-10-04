/**
 * Get the keys of an object. This is a wrapper around `Object.keys` that types the result as an array of the keys of the object.
 * @param obj The object to get the keys of.
 * @returns The keys of the object.
 */
export const objectKeys = <T extends Record<string, unknown>>(obj: T) => Object.keys(obj) as Array<keyof T>;
