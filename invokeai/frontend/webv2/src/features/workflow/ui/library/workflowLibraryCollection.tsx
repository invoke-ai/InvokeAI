import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { loadWorkflowLibraryRange, setWorkflowLibraryVisibleDemand } from '@features/workflow/data/libraryBrowseStore';
import { useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useMemo, useState } from 'react';

/**
 * What the grid and the list layouts share over one collection: duplicate-safe entries, the templates the project
 * holds copies of, and the server-window demand for the positions on screen.
 */

const NO_LINKED_TEMPLATES: ReadonlySet<string> = new Set();

/** Keep duplicate server identities from corrupting card and row reconciliation. */
export const dedupeWorkflowLibraryEntries = (entries: readonly WorkflowLibraryEntry[]): WorkflowLibraryEntry[] => {
  const seen = new Set<string>();
  const unique: WorkflowLibraryEntry[] = [];

  for (const entry of entries) {
    if (!seen.has(entry.item.workflow_id)) {
      seen.add(entry.item.workflow_id);
      unique.push(entry);
    }
  }

  return unique;
};

/** The library templates the project holds copies of, as one string so unrelated project edits keep it equal. */
const selectLinkedTemplateKey = (project: { workflows: readonly { source?: { libraryWorkflowId: string } }[] }) =>
  project.workflows
    .flatMap((workflow) => (workflow.source ? [workflow.source.libraryWorkflowId] : []))
    .sort()
    .join('\n');

/** Library templates the project holds a copy of; the project's own collection marks none. */
export const useLinkedLibraryTemplates = (isProjectCollection: boolean): ReadonlySet<string> => {
  const linkedTemplateKey = useWorkflowProjectSelector(selectLinkedTemplateKey);

  return useMemo(
    () => (isProjectCollection || !linkedTemplateKey ? NO_LINKED_TEMPLATES : new Set(linkedTemplateKey.split('\n'))),
    [isProjectCollection, linkedTemplateKey]
  );
};

export interface WorkflowLibraryDemandRange {
  /** First and last server positions to keep resident, overscan included. */
  first: number;
  last: number;
  /** The positions on screen, which enrichment reaches before either overscan edge. */
  visibleFirst: number;
  visibleLast: number;
}

/** Resident identities in the range: visible ones first, so the bounded enrichment slots reach them before overscan. */
export const getWorkflowLibraryDemandIds = (
  entries: readonly WorkflowLibraryEntry[],
  startIndex: number,
  { first, last, visibleFirst, visibleLast }: WorkflowLibraryDemandRange
): string[] => {
  const visible: string[] = [];
  const overscan: string[] = [];

  for (const [index, entry] of entries.entries()) {
    const position = startIndex + index;

    if (position >= first && position <= last) {
      const demand = position >= visibleFirst && position <= visibleLast ? visible : overscan;
      demand.push(entry.item.workflow_id);
    }
  }

  return visible.concat(overscan);
};

// Mount leases keep external-store writes out of render and release detail demand when the viewport leaves.
let demandLease = 0;

/**
 * Demands one window: key it by its range and identities, so each change remounts it. The newest mount owns the
 * visible demand; an unmount clears it only if no newer mount took over in the meantime.
 */
export const WorkflowLibraryDemand = ({ ids, start, end }: { ids: readonly string[]; start: number; end: number }) => {
  useMountEffect(() => {
    const lease = ++demandLease;
    setWorkflowLibraryVisibleDemand(ids);
    void loadWorkflowLibraryRange(start, end);
    return () => {
      queueMicrotask(() => {
        if (lease === demandLease) {
          setWorkflowLibraryVisibleDemand([]);
        }
      });
    };
  });
  return null;
};

/**
 * A demand key that stops following its window while the collection is hidden under the preview stage: the demand
 * keeps the positions it showed, so a preview stepping elsewhere in the results does not pull the window back, and
 * showing the collection again demands its window afresh only if that window changed meanwhile.
 */
export const useFrozenWhileHidden = (key: string, isHidden: boolean): string => {
  const [frozenKey, setFrozenKey] = useState<string | null>(null);

  if (isHidden && frozenKey === null) {
    setFrozenKey(key);
  } else if (!isHidden && frozenKey !== null) {
    setFrozenKey(null);
  }

  return (isHidden ? frozenKey : null) ?? key;
};
