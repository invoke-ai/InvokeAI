import { z } from 'zod';

import { zInvkFontDependency } from './fonts';
import { INVK_EXTENSION, INVK_VERSION, InvkFormatError } from './format';

/**
 * The v2 ZIP contains manifest.json, project.json, optional board.json and cover, and images/<name> or
 * videos/<name>. Recognize v1 only to return a specific unsupported-version refusal.
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
  fonts: z.array(zInvkFontDependency).optional(),
  /** Highest canvas schema version required by the source project and its retained history. */
  minimumCanvasSchemaVersion: z.number().int().min(1).optional(),
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

/** Accept v2; recognize v1 only for a named refusal. */
export const parseInvkManifest = (data: unknown): InvkManifest => {
  const parsed = zManifest.safeParse(data);

  if (!parsed.success) {
    const version = (data as { version?: unknown } | null)?.version;

    if (typeof version !== 'number') {
      throw new InvkFormatError('not-a-project');
    }

    // Report malformed known versions as corrupt, not newer-version incompatible.
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
  fonts?: InvkManifest['fonts'];
  minimumCanvasSchemaVersion?: number;
  name: string;
  sourceProjectId?: string;
}): InvkManifest => ({
  appVersion: input.appVersion,
  contents: 'workbench-project',
  createdAt: input.createdAt,
  name: input.name,
  version: INVK_VERSION,
  ...(input.fonts === undefined || input.fonts.length === 0 ? {} : { fonts: input.fonts }),
  ...(input.cover === undefined ? {} : { cover: input.cover }),
  ...(input.minimumCanvasSchemaVersion === undefined
    ? {}
    : { minimumCanvasSchemaVersion: input.minimumCanvasSchemaVersion }),
  ...(input.sourceProjectId === undefined ? {} : { sourceProjectId: input.sourceProjectId }),
});

/** Remove forbidden/control characters while preserving other Unicode filename characters. */
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
