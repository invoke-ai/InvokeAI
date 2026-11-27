import type { RefusedWorkbenchProject } from '@workbench/projectContracts';

import { MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';

/** Keep format/error constants eager and zod/ZIP dependencies lazy. */

export const INVK_EXTENSION = '.invk';
export const INVK_MIME_TYPE = 'application/zip';

/**
 * Version 3 is written: `project.json` is the project content, beside an optional `workspace.json`. Version 2, whose
 * `project.json` is a whole document, still imports; version 1 receives a named refusal. board.json is optional.
 */
export const INVK_VERSION = 3;

/** Fixed entry paths. `cover` varies by image format and is named in the manifest. */
export const INVK_MANIFEST_ENTRY = 'manifest.json';
export const INVK_DOCUMENT_ENTRY = 'project.json';

/** The exporter's workspace (layout, generation settings, prompt history), when they chose to include it. */
export const INVK_WORKSPACE_ENTRY = 'workspace.json';

/** The project board's contents; see `board.ts` for why a reader cannot infer them. */
export const INVK_BOARD_ENTRY = 'board.json';

/** Media uses kind-specific folders; readers tolerate unknown archive entries. */
export const INVK_IMAGES_PREFIX = 'images/';
export const INVK_VIDEOS_PREFIX = 'videos/';

export class FontImportQuotaError extends Error {
  constructor() {
    super('The font library has insufficient space for the embedded fonts.');
    this.name = 'FontImportQuotaError';
  }
}

export type InvkFormatReason =
  /** A ZIP, but the manifest is a canvas project written by the previous frontend. */
  | 'legacy-canvas-project'
  /** A ZIP with a manifest we recognize the shape of but not the version. */
  | 'unsupported-version'
  /** Not a project archive at all: not a ZIP, no manifest, or an unreadable one. */
  | 'not-a-project'
  /** A project archive whose payload is missing or will not rehydrate. */
  | 'damaged'
  /** The file is larger than the archive budget, or expands past it. */
  | 'too-large';

/** Callers localize typed failure reasons. */
export class InvkFormatError extends Error {
  readonly reason: InvkFormatReason;

  constructor(reason: InvkFormatReason, message = `Invalid .invk archive: ${reason}`) {
    super(message);
    this.name = 'InvkFormatError';
    this.reason = reason;
  }
}

/**
 * The archive error for a document the load refused. People see the reason ("made with a newer Invoke"); the message,
 * kept for logs and error details, names which schema was too new.
 */
export const toInvkRefusalError = (refused: RefusedWorkbenchProject): InvkFormatError => {
  if (refused.refusal.status !== 'unsupported-version') {
    return new InvkFormatError('damaged', 'The project document was refused.');
  }
  const { version } = refused.refusal;
  if (refused.source === 'project-document') {
    return new InvkFormatError('unsupported-version', `Project requires document schema ${version}.`);
  }
  return version < MIN_SUPPORTED_CANVAS_SCHEMA_VERSION
    ? new InvkFormatError('legacy-canvas-project', `Project uses retired canvas schema ${version}.`)
    : new InvkFormatError('unsupported-version', `Project requires canvas schema ${version}.`);
};
