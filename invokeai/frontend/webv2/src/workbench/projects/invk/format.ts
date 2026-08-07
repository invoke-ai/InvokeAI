/**
 * What an `.invk` is called and how reading one can fail.
 *
 * Split from `manifest.ts`, which owns the zod schemas, because these are needed *eagerly* — the
 * picker names the extension and every call site translates the error's reason. That is what keeps
 * zod and the ZIP codec behind the lazy import.
 */

export const INVK_EXTENSION = '.invk';
export const INVK_MIME_TYPE = 'application/zip';

/**
 * Workbench projects are version 2; version 1 is the previous frontend's canvas project, refused by
 * name. Board membership did not bump this — webv2 has not shipped, so no v2 archive predating
 * `board.json` exists outside dev builds, and an archive without the entry names no board.
 */
export const INVK_VERSION = 2;

/** Fixed entry paths. `cover` varies by image format and is named in the manifest. */
export const INVK_MANIFEST_ENTRY = 'manifest.json';
export const INVK_DOCUMENT_ENTRY = 'project.json';

/** The project board's contents; see `board.ts` for why a reader cannot infer them. */
export const INVK_BOARD_ENTRY = 'board.json';

/**
 * Filed by kind, not pooled: import must know which namespace an entry belongs to, and a folder
 * states it where a shared one would leave it to a file extension. `images/` is byte-identical to
 * the legacy v1 container. `videos/` needed no version bump — `readInvkArchive` ignores entries it
 * does not recognize, so an older reader just leaves those references dangling.
 */
export const INVK_IMAGES_PREFIX = 'images/';
export const INVK_VIDEOS_PREFIX = 'videos/';

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

/**
 * Every way reading an `.invk` can fail, as a value rather than a message. Call
 * sites map `reason` to a translated string; nothing user-facing is built here.
 */
export class InvkFormatError extends Error {
  readonly reason: InvkFormatReason;

  constructor(reason: InvkFormatReason, message = `Invalid .invk archive: ${reason}`) {
    super(message);
    this.name = 'InvkFormatError';
    this.reason = reason;
  }
}
