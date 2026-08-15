import { z } from 'zod';

import { INVK_MAX_ENTRIES } from './archive';
import { InvkFormatError } from './format';

/**
 * `board.json`: what was on the project's board when the archive was written. A document names only
 * the media it draws with, so everything else the project produced appears nowhere in it — this
 * entry is the difference between carrying a project's canvas and carrying its workspace.
 *
 * ```json
 * { "version": 1, "items": [{ "category": "general", "kind": "image", "name": "…", "starred": false }] }
 * ```
 *
 * The version is the *file's*, not the API's: an archive format pinned to the wire format would
 * have to move whenever it did, and vice versa.
 *
 * Validation is strict and structural — a duplicate `(kind, name)` or a non-basename name means the
 * file is malformed and must be refused before anything is uploaded, not reported alongside it.
 */

export type InvkMediaKind = 'image' | 'video';
export type InvkMediaCategory = 'general' | 'control' | 'mask' | 'user';

export interface InvkBoardItem {
  category: InvkMediaCategory;
  kind: InvkMediaKind;
  name: string;
  starred: boolean;
}

export interface InvkBoardSnapshot {
  version: 1;
  items: InvkBoardItem[];
}

/**
 * A media name must be exactly what the server would name a file: no separators, no traversal, no
 * NUL. The archive stores these under `images/` and `videos/`, so anything else would either
 * escape that prefix or fail to round-trip through the ZIP.
 */
const zMediaName = z
  .string()
  .min(1)
  .refine((name) => !name.includes('/') && !name.includes('\\') && !name.includes('\0'), {
    message: 'must be a basename',
  })
  .refine((name) => name !== '.' && name !== '..', { message: 'must name a file' });

const zBoardItem = z.object({
  category: z.enum(['general', 'control', 'mask', 'user']),
  kind: z.enum(['image', 'video']),
  name: zMediaName,
  starred: z.boolean(),
});

const zBoardSnapshot = z
  .object({
    // Bounded by the same ceiling as archive entries: every item is at most one entry, so a list
    // that could not be packed cannot be honest either.
    items: z.array(zBoardItem).max(INVK_MAX_ENTRIES),
    version: z.literal(1),
  })
  // Unknown keys are refused rather than ignored: this file is small, entirely generated, and a
  // reader silently dropping a field a later version added would restore the wrong thing quietly.
  .strict();

/** Sort by kind then name. Images and videos are separate namespaces, so one name can be both. */
const compareItems = (left: InvkBoardItem, right: InvkBoardItem): number =>
  left.kind.localeCompare(right.kind) || left.name.localeCompare(right.name);

/**
 * Parse and canonicalize `board.json`.
 *
 * Input order is free — the server's ordering is its own business — but the result is sorted, so
 * every consumer sees one order and tests can compare whole structures.
 */
export const parseInvkBoardSnapshot = (data: unknown): InvkBoardSnapshot => {
  const parsed = zBoardSnapshot.safeParse(data);

  if (!parsed.success) {
    throw new InvkFormatError('damaged', `Invalid ${'board.json'}: ${parsed.error.issues[0]?.message ?? 'malformed'}`);
  }

  const seen = new Set<string>();

  for (const item of parsed.data.items) {
    const key = `${item.kind}:${item.name}`;

    if (seen.has(key)) {
      throw new InvkFormatError('damaged', `Duplicate board entry ${key}`);
    }

    seen.add(key);
  }

  return { items: [...parsed.data.items].sort(compareItems), version: 1 };
};

/** Build the entry from a server snapshot. Sorting here is what makes exports byte-comparable. */
export const buildInvkBoardSnapshot = (items: readonly InvkBoardItem[]): InvkBoardSnapshot => ({
  items: [...items].sort(compareItems),
  version: 1,
});
