import { z } from 'zod';

import { INVK_EXTENSION, INVK_VERSION, InvkFormatError } from './format';

/**
 * The `.invk` manifest. The previous frontend's canvas project files are ZIPs with this same root
 * `manifest.json` and `images/` folder; that container is kept verbatim, and version 2 changes only
 * the payload.
 *
 * ```
 * <name>.invk
 * ├── manifest.json          this file's shape
 * ├── project.json           the project document
 * ├── board.json             the project board's visible contents
 * ├── cover.<ext>            optional preview; the entry name is recorded here
 * ├── images/<image_name>    bytes, named exactly as the server names them
 * └── videos/<video_name>    the same, for the other namespace
 * ```
 *
 * Parsed as a discriminated union rather than `z.literal(2)` so a refusal can name the version
 * precisely: "this is a canvas project from an earlier version" is actionable, a zod issue list is
 * not.
 *
 * Adding `board.json` did not bump the version — a version tells a reader what it may assume about
 * a file *someone else wrote*, and webv2 has not shipped. Readers treat the entry as optional.
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

const zManifest = z.discriminatedUnion('version', [zManifestV1, zManifestV2]);

/** Every version this app has ever written or can name. Anything else genuinely is from the future. */
const KNOWN_VERSIONS: ReadonlySet<number> = new Set([1, 2]);

/** A manifest this app can read: the workbench project container. */
export type InvkManifest = z.infer<typeof zManifestV2>;

/**
 * Accepts a v2 manifest; throws {@link InvkFormatError} for anything else.
 * Never returns a v1 manifest — recognizing v1 exists only to name the refusal.
 */
export const parseInvkManifest = (data: unknown): InvkManifest => {
  const parsed = zManifest.safeParse(data);

  if (!parsed.success) {
    const version = (data as { version?: unknown } | null)?.version;

    if (typeof version !== 'number') {
      throw new InvkFormatError('not-a-project');
    }

    // A version this app has written is a version it can read, so a manifest that still fails to
    // parse at one of them is damaged — a truncated name, a missing timestamp. Calling that
    // "written by a newer version of Invoke" tells someone to go and upgrade over a broken file.
    throw new InvkFormatError(KNOWN_VERSIONS.has(version) ? 'damaged' : 'unsupported-version');
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
