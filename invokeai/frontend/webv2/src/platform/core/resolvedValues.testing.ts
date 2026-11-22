import type { ResolvedValues } from './resolvedValues';

export const resolvedValues = (values: Record<string, unknown> = {}): ResolvedValues => values as ResolvedValues;
