import type { MigrationException } from './dependencyPolicy';

/** Temporary exceptions require an open removal ticket; the completion gate requires none. */
export const migrationExceptions: readonly MigrationException[] = [] as const;
