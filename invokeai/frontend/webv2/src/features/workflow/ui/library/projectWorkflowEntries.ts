import type { InvocationTemplates, ProjectWorkflowEntry, ProjectWorkflowSource } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { galleryImageUrls } from '@features/gallery/utility';
import { parseWorkflowTags } from '@features/workflow/core/libraryTags';
import { extractWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';

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

/** Bundled template ids are a server invariant (`default_` prefix); they are read-only whoever asks. */
export const isBundledLibraryWorkflowId = (libraryWorkflowId: string): boolean =>
  libraryWorkflowId.startsWith('default_');

/** Whether a source can be written to at all; the server still has the final word (ownership, deletion). */
export const isUpdatableSource = (source: ProjectWorkflowSource | undefined): source is ProjectWorkflowSource =>
  source !== undefined && !isBundledLibraryWorkflowId(source.libraryWorkflowId);

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
