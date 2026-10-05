import { z } from 'zod';

import { INVK_MAX_ENTRIES } from './archive';
import { InvkFormatError } from './format';

/**
 * The independently versioned board entry includes unreferenced media. Validate it strictly before uploading
 * anything.
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

/** Names must be basenames without separators, traversal, or NUL to round-trip safely as archive paths. */
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
    // Board item count cannot exceed the archive entry capacity.
    items: z.array(zBoardItem).max(INVK_MAX_ENTRIES),
    version: z.literal(1),
  })
  // Reject unknown fields rather than silently discard future semantics.
  .strict();

/** Sort by kind then name. Images and videos are separate namespaces, so one name can be both. */
const compareItems = (left: InvkBoardItem, right: InvkBoardItem): number =>
  left.kind.localeCompare(right.kind) || left.name.localeCompare(right.name);

/** Accept arbitrary input order; emit deterministic order. */
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
