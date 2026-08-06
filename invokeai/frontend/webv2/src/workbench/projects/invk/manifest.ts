import { z } from 'zod';

import { INVK_EXTENSION, INVK_VERSION, InvkFormatError } from './format';

/**
 * The `.invk` manifest: what an archive is, before anything reads what is in it.
 *
 * `.invk` is not new. The canvas project files written by the previous frontend
 * are ZIPs with this same root `manifest.json` — `{version, appVersion,
 * createdAt, name}` — alongside an `images/` folder keyed by backend image name.
 * That container is kept verbatim. Version 2 changes only the payload: where a
 * v1 archive carried four canvas-shaped state files, a v2 archive carries one
 * `project.json` holding the whole workbench project document, plus an optional
 * cover. Version 3 adds `board.json`, the project board's contents — the part of
 * a project that the document does not reference and a reader cannot infer.
 *
 * ```
 * <name>.invk
 * ├── manifest.json          this file's shape
 * ├── project.json           the project document (v2 and v3)
 * ├── board.json             the project board's visible contents (v3 only)
 * ├── cover.<ext>            optional preview; the entry name is recorded here
 * ├── images/<image_name>    bytes, named exactly as the server names them
 * └── videos/<video_name>    the same, for the other namespace
 * ```
 *
 * Parsing is a discriminated union rather than a strict `z.literal(3)` so that
 * every version this app has written can be named precisely when refused. "This
 * is a canvas project from an earlier version" is a sentence someone can act on;
 * a zod issue list is not. The legacy reader is the mirror image — it pins
 * `version: 1` and will refuse the others, which is correct and needs no change.
 *
 * Version 2 is still read, and is deliberately not asked to carry board
 * membership: an archive written before boards existed has no board to describe,
 * and inventing one for it would claim knowledge the file does not contain.
 *
 * The names and the error class live in `./format`, which carries no zod, so
 * that the file picker and the error toast do not drag the schema into the
 * Launchpad's initial bundle.
 */

const zManifestV1 = z.object({
  appVersion: z.string(),
  createdAt: z.string(),
  name: z.string(),
  version: z.literal(1),
});

const zManifestV2 = z.object({
  appVersion: z.string(),
  /** Discriminates a workbench project from whatever a later version may carry. */
  contents: z.literal('workbench-project'),
  /** Entry path of the preview image, when the archive carries one. */
  cover: z.string().optional(),
  createdAt: z.string(),
  name: z.string(),
  /** The project this was exported from. Informational — import always mints a fresh id. */
  sourceProjectId: z.string().optional(),
  version: z.literal(2),
});

const zManifestV3 = zManifestV2.extend({ version: z.literal(3) });

const zManifest = z.discriminatedUnion('version', [zManifestV1, zManifestV2, zManifestV3]);

/** A manifest this app can read: v2 or v3, which differ only in what else the archive holds. */
export type InvkManifest = z.infer<typeof zManifestV2> | z.infer<typeof zManifestV3>;

/**
 * Accepts a v2 or v3 manifest; throws {@link InvkFormatError} for anything else.
 * Never returns a v1 manifest — recognizing v1 exists only to name the refusal.
 */
export const parseInvkManifest = (data: unknown): InvkManifest => {
  const parsed = zManifest.safeParse(data);

  if (!parsed.success) {
    const version = (data as { version?: unknown } | null)?.version;

    throw new InvkFormatError(typeof version === 'number' ? 'unsupported-version' : 'not-a-project');
  }

  if (parsed.data.version === 1) {
    throw new InvkFormatError('legacy-canvas-project');
  }

  return parsed.data;
};

export const buildInvkManifest = (input: {
  appVersion: string;
  cover?: string;
  createdAt: string;
  name: string;
  sourceProjectId?: string;
}): InvkManifest => ({
  appVersion: input.appVersion,
  contents: 'workbench-project',
  createdAt: input.createdAt,
  name: input.name,
  version: INVK_VERSION,
  ...(input.cover === undefined ? {} : { cover: input.cover }),
  ...(input.sourceProjectId === undefined ? {} : { sourceProjectId: input.sourceProjectId }),
});

/**
 * File name for the download. Replaces what filesystems reject and drops
 * control/format characters, but leaves the rest of unicode alone — a project
 * named in a non-Latin script should not export as `project.invk`.
 */
export const toInvkFileName = (projectName: string): string => {
  const trimmed = projectName
    .replaceAll(/["*/:<>?\\|]/gu, '_')
    .replaceAll(/\p{C}/gu, '')
    .trim()
    // Windows cannot open a name ending in a dot, and a leading one hides the file.
    .replace(/\.+$/u, '')
    .replace(/^\.+/u, '')
    .trim();

  return `${trimmed || 'project'}${INVK_EXTENSION}`;
};
