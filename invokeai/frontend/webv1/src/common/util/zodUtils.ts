import type { z } from 'zod';

/**
 * Helper to create a type guard from a zod schema. The type guard will infer the schema's TS type.
 * @param schema The zod schema to create a type guard from.
 * @returns A type guard function for the schema.
 */
export const buildZodTypeGuard = <T extends z.ZodTypeAny>(schema: T) => {
  return (val: unknown): val is z.infer<T> => schema.safeParse(val).success;
};
