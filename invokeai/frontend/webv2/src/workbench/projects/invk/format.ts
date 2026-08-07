/**
 * What an `.invk` is called and how reading one can fail.
 *
 * Split from `manifest.ts`, which owns the zod schemas, because these few
 * constants and the error class are needed *eagerly*: the file picker names the
 * extension, and every import call site catches the error to translate its
 * reason. The schema is needed only once someone has actually chosen a file.
 * Keeping them apart is what lets the picker and the toast stay in the
 * Launchpad's initial graph while zod and the parser stay behind the lazy
 * import that pulls in the ZIP codec.
 */

export const INVK_EXTENSION = '.invk';
export const INVK_MIME_TYPE = 'application/zip';

/**
 * Workbench projects are version 2 of this container; version 1 is the previous frontend's canvas
 * project, which is refused by name.
 *
 * Board membership did not bump this. A version exists to tell a reader what it is allowed to
 * assume about a file *someone else wrote*, and webv2 has not shipped — no v2 archive exists
 * anywhere that predates `board.json`, so a version that distinguished them would describe a
 * population of files with no members. An archive without the entry is still read (dev builds wrote
 * some), and it means what its absence says: this file names no board.
 */
export const INVK_VERSION = 2;

/** Fixed entry paths. `cover` varies by image format and is named in the manifest. */
export const INVK_MANIFEST_ENTRY = 'manifest.json';
export const INVK_DOCUMENT_ENTRY = 'project.json';

/**
 * The project board's contents, as the exporting server enumerated them.
 *
 * A project document only references the media it draws with. Everything else the project produced
 * — results generated but never placed on canvas — is on its board and named nowhere in the
 * document, so a reader has no way to infer it. This entry states it, which is what makes an
 * `.invk` the project's workspace rather than only its canvas.
 */
export const INVK_BOARD_ENTRY = 'board.json';

/**
 * Bundled bytes are filed by kind, not pooled. Images and videos are separate
 * backend namespaces with separate fetch and upload routes, so import has to
 * know which an entry is before it can restore it — and a folder states that
 * where a shared one would leave it to be guessed from a file extension.
 *
 * `images/` is also byte-identical to the legacy v1 container, which is the
 * continuity this format was built around. Adding `videos/` needs no version
 * bump: `readInvkArchive` ignores entries it does not recognize, so a reader
 * predating videos reads such an archive and simply leaves those references
 * dangling — the same degradation an absent image already gets.
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
