import { sha256Hex } from '@platform/browser/sha256';
import { z } from 'zod';

import { InvkFormatError } from './format';

const zFontReference = z.object({
  contentHash: z.string().regex(/^[a-f0-9]{64}$/u),
  family: z.string(),
  id: z.string().min(1),
  label: z.string(),
});

export const zInvkFontDependency = z.object({
  contentHash: z.string().regex(/^[a-f0-9]{64}$/u),
  entry: z
    .string()
    .regex(/^fonts\/[a-f0-9]{64}\.(ttf|otf|woff|woff2)$/u)
    .optional(),
  family: z.string(),
  label: z.string(),
  references: z.array(z.string().min(1)).min(1),
});

export type InvkFontReference = z.infer<typeof zFontReference>;
export type InvkFontDependency = z.infer<typeof zInvkFontDependency>;
/** Matches the backend's hard ceiling; the receiving server can configure a lower upload limit. */
export const INVK_MAX_FONT_BYTES = 128 * 1024 * 1024;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** Only text sources own font references; arbitrary workflow fields are not font dependencies. */
const walkTextSources = (value: unknown, visit: (source: Record<string, unknown>) => void): void => {
  if (Array.isArray(value)) {
    for (const item of value) {
      walkTextSources(item, visit);
    }
  } else if (isRecord(value)) {
    if (value.type === 'text' && 'fontRef' in value) {
      visit(value);
      return;
    }
    for (const child of Object.values(value)) {
      walkTextSources(child, visit);
    }
  }
};

export const collectFontDependencies = (document: Record<string, unknown>): InvkFontDependency[] => {
  const byHash = new Map<string, InvkFontDependency>();
  walkTextSources(document.canvas, (source) => {
    const parsed = zFontReference.safeParse(source.fontRef);
    if (!parsed.success) {
      throw new InvkFormatError('damaged', 'A text layer has an invalid font reference.');
    }
    const ref = parsed.data;
    const dependency = byHash.get(ref.contentHash);
    if (dependency) {
      if (!dependency.references.includes(ref.id)) {
        dependency.references.push(ref.id);
      }
    } else {
      byHash.set(ref.contentHash, {
        contentHash: ref.contentHash,
        family: ref.family,
        label: ref.label,
        references: [ref.id],
      });
    }
  });
  return [...byHash.values()].sort((a, b) => a.contentHash.localeCompare(b.contentHash));
};

export const remapFontReferences = (
  document: Record<string, unknown>,
  byHash: ReadonlyMap<string, InvkFontReference>
): Record<string, unknown> => {
  if (byHash.size === 0) {
    return document;
  }
  const copy = structuredClone(document);
  walkTextSources(copy.canvas, (source) => {
    const ref = zFontReference.parse(source.fontRef);
    const replacement = byHash.get(ref.contentHash);
    if (replacement) {
      source.fontRef = replacement;
    }
  });
  return copy;
};

export interface EmbeddedFont {
  bytes: Uint8Array;
  dependency: InvkFontDependency;
  filename: string;
}

const hasExactReferences = (expected: readonly string[], actual: readonly string[]): boolean => {
  if (expected.length !== actual.length) {
    return false;
  }

  const actualSet = new Set(actual);

  return actualSet.size === actual.length && expected.every((reference) => actualSet.has(reference));
};

/** Checks the dependency manifest against the document and authenticates every embedded payload. */
export const readEmbeddedFonts = async (
  dependencies: readonly InvkFontDependency[],
  document: Record<string, unknown>,
  entries: ReadonlyMap<string, Uint8Array>
): Promise<EmbeddedFont[]> => {
  const required = new Map(collectFontDependencies(document).map((dependency) => [dependency.contentHash, dependency]));
  const byHash = new Map(dependencies.map((dependency) => [dependency.contentHash, dependency]));
  const declaredEntries = new Set(dependencies.flatMap((dependency) => (dependency.entry ? [dependency.entry] : [])));
  if (byHash.size !== dependencies.length) {
    throw new InvkFormatError('damaged', 'Duplicate font dependencies.');
  }
  if (required.size !== byHash.size || [...required.keys()].some((contentHash) => !byHash.has(contentHash))) {
    throw new InvkFormatError('damaged', 'Font dependencies do not match the project.');
  }
  const embedded: EmbeddedFont[] = [];
  for (const dependency of dependencies) {
    const expected = required.get(dependency.contentHash);
    if (!expected || !hasExactReferences(expected.references, dependency.references)) {
      throw new InvkFormatError('damaged', 'Font dependencies do not match the project.');
    }
    if (!dependency.entry) {
      continue;
    }
    const bytes = entries.get(dependency.entry);
    if (
      !dependency.entry.startsWith(`fonts/${dependency.contentHash}.`) ||
      !bytes ||
      bytes.byteLength > INVK_MAX_FONT_BYTES ||
      (await sha256Hex(bytes)) !== dependency.contentHash
    ) {
      throw new InvkFormatError('damaged', `Invalid embedded font: ${dependency.label}`);
    }
    embedded.push({ bytes, dependency, filename: dependency.entry.slice('fonts/'.length) });
  }
  for (const path of entries.keys()) {
    if (path.startsWith('fonts/') && !declaredEntries.has(path)) {
      throw new InvkFormatError('damaged', 'An embedded font is not declared in the manifest.');
    }
  }
  return embedded;
};

export interface FontArchiveTransport {
  download: (dependency: InvkFontDependency, signal?: AbortSignal) => Promise<{ bytes: Uint8Array; filename: string }>;
  validate: (file: File, signal?: AbortSignal) => Promise<void>;
  upload: (file: File, signal?: AbortSignal) => Promise<{ created: boolean; font: InvkFontReference }>;
  remove: (id: string, signal?: AbortSignal) => Promise<void>;
}

export interface RestoredFontLedger {
  createdIds: string[];
  mappings: Map<string, InvkFontReference>;
}

export const createRestoredFontLedger = (): RestoredFontLedger => ({ createdIds: [], mappings: new Map() });

const embeddedFile = (font: EmbeddedFont): File =>
  new File([font.bytes as BlobPart], font.filename, { type: 'application/octet-stream' });

export const preflightEmbeddedFonts = async (
  fonts: readonly EmbeddedFont[],
  transport: FontArchiveTransport,
  signal?: AbortSignal
): Promise<void> => {
  // Validate all files before the first upload, including files later deduplicated by the server.
  for (const font of fonts) {
    signal?.throwIfAborted();
    await transport.validate(embeddedFile(font), signal);
  }
};

export const restoreEmbeddedFonts = async (
  fonts: readonly EmbeddedFont[],
  ledger: RestoredFontLedger,
  transport: FontArchiveTransport,
  signal?: AbortSignal,
  onProgress?: (completed: number, total: number) => void
): Promise<void> => {
  let completed = 0;
  onProgress?.(completed, fonts.length);
  for (const font of fonts) {
    signal?.throwIfAborted();
    const result = await transport.upload(embeddedFile(font), signal);
    if (result.created) {
      ledger.createdIds.push(result.font.id);
    }
    if (result.font.contentHash !== font.dependency.contentHash) {
      throw new InvkFormatError('damaged', 'The uploaded font does not match its archive checksum.');
    }
    ledger.mappings.set(font.dependency.contentHash, result.font);
    completed += 1;
    onProgress?.(completed, fonts.length);
  }
};

export const rollbackRestoredFonts = async (
  ledger: RestoredFontLedger,
  transport: FontArchiveTransport,
  signal?: AbortSignal
): Promise<void> => {
  for (const id of ledger.createdIds) {
    signal?.throwIfAborted();
    try {
      await transport.remove(id, signal);
    } catch {
      // Best effort: cleanup must not mask the import failure or delete pre-existing resources.
    }
  }
  ledger.createdIds = [];
  ledger.mappings.clear();
};
