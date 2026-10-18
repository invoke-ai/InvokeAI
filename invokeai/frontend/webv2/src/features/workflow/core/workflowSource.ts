import type { ProjectWorkflowEntry, ProjectWorkflowSource } from './types';

/**
 * What a project workflow's library source allows, decided from the source alone. A leaf on purpose: the widget
 * chrome and the copy-choice flows ask at boot, and must not carry the requirement extraction the library cards need.
 */

/** Bundled template ids are a server invariant (`default_` prefix); they are read-only whoever asks. */
export const isBundledLibraryWorkflowId = (libraryWorkflowId: string): boolean =>
  libraryWorkflowId.startsWith('default_');

/** Whether a source can be written to at all; the server still has the final word (ownership, deletion). */
export const isUpdatableSource = (source: ProjectWorkflowSource | undefined): source is ProjectWorkflowSource =>
  source !== undefined && !isBundledLibraryWorkflowId(source.libraryWorkflowId);

/** The project copies of one library template, in collection order. */
export const findProjectCopiesOf = (
  workflows: readonly ProjectWorkflowEntry[],
  libraryWorkflowId: string
): ProjectWorkflowEntry[] => workflows.filter((entry) => entry.source?.libraryWorkflowId === libraryWorkflowId);
