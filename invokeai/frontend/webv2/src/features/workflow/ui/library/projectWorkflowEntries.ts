import type { AuthSession } from '@features/identity';
import type { InvocationTemplates, ProjectWorkflowEntry, ProjectWorkflowSource } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry, WorkflowTagCount } from '@features/workflow/data/libraryBrowseStore';
import type { ProjectWorkflowOrder } from '@features/workflow/ui/workflowUiStore';

import { galleryImageUrls } from '@features/gallery/utility';
import { parseWorkflowTags, sortTagCounts } from '@features/workflow/core/libraryTags';
import { extractWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { isBundledLibraryWorkflowId } from '@features/workflow/queries';

/**
 * A project workflow presented through the library's card and detail contracts. The mapping is cached per entry
 * identity, so editing one workflow re-derives one card and leaves every other card's props untouched.
 */

export interface ProjectWorkflowLibraryEntry extends WorkflowLibraryEntry {
  /** The project workflow the card stands for. */
  projectWorkflow: ProjectWorkflowEntry;
}

const cache = new WeakMap<
  ProjectWorkflowEntry,
  { entry: ProjectWorkflowLibraryEntry; templates: InvocationTemplates | null }
>();

/** Whether a source can be written to at all; the server still has the final word (ownership, deletion). */
export const isUpdatableSource = (source: ProjectWorkflowSource | undefined, session: AuthSession): boolean =>
  source !== undefined &&
  session.phase === 'ready' &&
  !isBundledLibraryWorkflowId(source.libraryWorkflowId) &&
  (!session.multiuserEnabled ||
    session.user?.is_admin === true ||
    (Boolean(session.user?.user_id) && source.ownerUserId === session.user?.user_id));

export const toProjectWorkflowLibraryEntry = (
  projectWorkflow: ProjectWorkflowEntry,
  templates: InvocationTemplates | null
): ProjectWorkflowLibraryEntry => {
  const cached = cache.get(projectWorkflow);

  if (cached && cached.templates === templates) {
    return cached.entry;
  }

  const { document, lastRun } = projectWorkflow;
  const entry: ProjectWorkflowLibraryEntry = {
    enrichment: templates
      ? {
          document,
          nodeCount: document.nodes.length,
          requirements: extractWorkflowModelRequirements(document, templates),
          status: 'ready',
        }
      : { status: 'pending' },
    item: {
      category: 'user',
      description: document.description,
      last_run_at: lastRun?.completedAt ?? null,
      name: document.name,
      revision: 0,
      tags: document.tags,
      thumbnail_url: lastRun ? galleryImageUrls.thumbnail(lastRun.imageName) : null,
      updated_at: document.updatedAt,
      workflow_id: document.id,
    },
    projectWorkflow,
    tags: parseWorkflowTags(document.tags),
  };

  cache.set(projectWorkflow, { entry, templates });

  return entry;
};

export const toProjectWorkflowLibraryEntries = (
  workflows: readonly ProjectWorkflowEntry[],
  templates: InvocationTemplates | null
): ProjectWorkflowLibraryEntry[] => workflows.map((workflow) => toProjectWorkflowLibraryEntry(workflow, templates));

/** The project copies of one library template, in collection order. */
export const findProjectCopiesOf = (
  workflows: readonly ProjectWorkflowEntry[],
  libraryWorkflowId: string
): ProjectWorkflowEntry[] => workflows.filter((entry) => entry.source?.libraryWorkflowId === libraryWorkflowId);

/**
 * How many of the project's workflows carry each tag, most used first. Case variants are one tag, labelled with the
 * casing seen first, and a workflow listing a tag twice counts once.
 */
export const countProjectWorkflowTags = (workflows: readonly ProjectWorkflowEntry[]): WorkflowTagCount[] => {
  const counts = new Map<string, WorkflowTagCount>();

  for (const { document } of workflows) {
    const seen = new Set<string>();

    for (const tag of parseWorkflowTags(document.tags)) {
      const key = tag.toLocaleLowerCase();

      if (!seen.has(key)) {
        seen.add(key);
        const existing = counts.get(key);
        counts.set(key, { count: (existing?.count ?? 0) + 1, tag: existing?.tag ?? tag });
      }
    }
  }

  return sortTagCounts([...counts.values()]);
};

export interface ProjectWorkflowQuery {
  /** Matched case-insensitively within names, descriptions and tags, as the server matches library searches. */
  search: string;
  /** Any-of, compared case-insensitively. */
  tags: readonly string[];
  order: ProjectWorkflowOrder;
}

const compareText = (left: string, right: string) => left.localeCompare(right, undefined, { sensitivity: 'base' });
/** ISO instants order as plain strings. */
const compareInstants = (left: string, right: string) => (left < right ? -1 : left > right ? 1 : 0);

/**
 * The project's workflows filtered and ordered like the library places, but locally. Workflows that never ran come
 * after the runs in either direction, as the server orders a missing `last_run_at`. Ties keep project order.
 */
export const queryProjectWorkflowEntries = <Entry extends ProjectWorkflowLibraryEntry>(
  entries: readonly Entry[],
  { order, search, tags }: ProjectWorkflowQuery
): readonly Entry[] => {
  const needle = search.trim().toLocaleLowerCase();
  const wanted = new Set(tags.map((tag) => tag.toLocaleLowerCase()));
  const filtered =
    needle || wanted.size > 0
      ? entries.filter(
          ({ item, tags: entryTags }) =>
            (!needle ||
              item.name.toLocaleLowerCase().includes(needle) ||
              item.description.toLocaleLowerCase().includes(needle) ||
              entryTags.some((tag) => tag.toLocaleLowerCase().includes(needle))) &&
            (wanted.size === 0 || entryTags.some((tag) => wanted.has(tag.toLocaleLowerCase())))
        )
      : entries;
  const sign = order.direction === 'asc' ? 1 : -1;

  switch (order.sort) {
    case 'order':
      return order.direction === 'asc' ? filtered : [...filtered].reverse();
    case 'name':
      return [...filtered].sort((left, right) => sign * compareText(left.item.name, right.item.name));
    case 'updated':
      return [...filtered].sort(
        (left, right) => sign * compareInstants(left.item.updated_at ?? '', right.item.updated_at ?? '')
      );
    case 'lastRun':
      return [...filtered].sort(({ item: { last_run_at: left } }, { item: { last_run_at: right } }) =>
        left && right ? sign * compareInstants(left, right) : Number(!left) - Number(!right)
      );
  }
};
