import type { LogEntry } from '@platform/logging/contracts';

import { LOG_LEVEL_ORDER, matchesLogScope } from '@platform/logging/contracts';
import { getLogSnapshot, subscribeLogs } from '@platform/logging/logger';
import { useCallback, useSyncExternalStore } from 'react';

/** Warnings and above for the project plus application events. */
export const countProblems = (entries: readonly LogEntry[], projectId: string): number => {
  const scope = { kind: 'project-and-application', projectId } as const;
  let count = 0;

  for (const entry of entries) {
    if (LOG_LEVEL_ORDER[entry.level] >= LOG_LEVEL_ORDER.warn && matchesLogScope(entry, scope)) {
      count += 1;
    }
  }

  return count;
};

/** The widget chip and header badge share this count. */
export const useProblemCount = (projectId: string): number => {
  const read = useCallback(() => countProblems(getLogSnapshot().entries, projectId), [projectId]);

  return useSyncExternalStore(subscribeLogs, read, read);
};
