import type { MigrationException } from './dependencyPolicy';

/**
 * Temporary, owned migration debt. The completion gate requires this list to be
 * empty; entries may only point at an open architecture-wayfinder removal ticket.
 */
export const migrationExceptions: readonly MigrationException[] = [] as const;
