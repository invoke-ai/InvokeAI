import type { RefusedWorkbenchProject } from '@workbench/projectContracts';

import { MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';

/** Keep format/error constants eager and zod/ZIP dependencies lazy. */

export const INVK_EXTENSION = '.invk';
export const INVK_MIME_TYPE = 'application/zip';

/** Version 2 is supported. Version 1 receives a named refusal; board.json is optional in v2. */
export const INVK_VERSION = 2;

/** Fixed entry paths. `cover` varies by image format and is named in the manifest. */
export const INVK_MANIFEST_ENTRY = 'manifest.json';
export const INVK_DOCUMENT_ENTRY = 'project.json';

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

/** The archive-format reason that matches a project the canvas version gate refused. */
export const toInvkFormatReason = (refused: RefusedWorkbenchProject): InvkFormatReason => {
  if (refused.refusal.status !== 'unsupported-version') {
    return 'damaged';
  }
  return refused.refusal.version < MIN_SUPPORTED_CANVAS_SCHEMA_VERSION
    ? 'legacy-canvas-project'
    : 'unsupported-version';
};
