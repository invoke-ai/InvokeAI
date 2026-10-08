import { z } from 'zod';

import { INVK_MAX_ENTRIES } from './archive';
import { InvkFormatError } from './format';

/**
 * The independently versioned board entry includes unreferenced media. Validate it strictly before uploading
 * anything.
 *
 * Version 2 carries every board of the project: the inbox first, then the boards made inside it. Version 1 carried
 * the inbox alone, as a flat item list; it is still read, as a project with no other boards.
 */

export type InvkMediaKind = 'image' | 'video';
export type InvkMediaCategory = 'general' | 'control' | 'mask' | 'user';

export interface InvkBoardItem {
  category: InvkMediaCategory;
  kind: InvkMediaKind;
  name: string;
  starred: boolean;
}

export interface InvkBoard {
  /** Whether the board is archived; restored as such, since an archived board is still the user's. */
  archived: boolean;
  /** The project's inbox, which the destination mints or claims rather than creates by name. */
  isInbox: boolean;
  items: InvkBoardItem[];
  /** The board's name. For the inbox it is informational: the inbox takes the project's name. */
  name: string;
}

export interface InvkBoardSnapshot {
  version: 2;
  /** The inbox first, then the project's other boards in the order the source listed them. */
  boards: InvkBoard[];
}

/** The longest board name the backend accepts. */
const MAX_BOARD_NAME_LENGTH = 300;

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

const zBoardSnapshotV1 = z
  .object({
    // Board item count cannot exceed the archive entry capacity.
    items: z.array(zBoardItem).max(INVK_MAX_ENTRIES),
    version: z.literal(1),
  })
  // Reject unknown fields rather than silently discard future semantics.
  .strict();

const zBoard = z
  .object({
    archived: z.boolean(),
    isInbox: z.boolean(),
    items: z.array(zBoardItem).max(INVK_MAX_ENTRIES),
    name: z.string().max(MAX_BOARD_NAME_LENGTH),
  })
  .strict();

/**
 * Boards an archive may name. Each one costs the importer a create and a move, so a file must not be able to ask
 * for an unbounded number of them in a few kilobytes; no real project comes near this.
 */
export const INVK_MAX_BOARDS = 1000;

const zBoardSnapshotV2 = z
  .object({
    boards: z.array(zBoard).min(1).max(INVK_MAX_BOARDS),
    version: z.literal(2),
  })
  .strict();

const zBoardSnapshot = z.union([zBoardSnapshotV1, zBoardSnapshotV2]);
const KNOWN_VERSIONS: ReadonlySet<number> = new Set([1, 2]);

/** Sort by kind then name. Images and videos are separate namespaces, so one name can be both. */
const compareItems = (left: InvkBoardItem, right: InvkBoardItem): number =>
  left.kind.localeCompare(right.kind) || left.name.localeCompare(right.name);

/** Every item of every board, in board order. An item is on one board, so this is a set. */
export const flattenInvkBoardItems = (snapshot: InvkBoardSnapshot): InvkBoardItem[] =>
  snapshot.boards.flatMap((board) => board.items);

/** Accept arbitrary input order; emit deterministic order. */
export const parseInvkBoardSnapshot = (data: unknown): InvkBoardSnapshot => {
  const parsed = zBoardSnapshot.safeParse(data);

  if (!parsed.success) {
    const version = (data as { version?: unknown } | null)?.version;

    // A board.json this reader does not know is a newer app's, not a broken file; a known version that fails is.
    if (typeof version === 'number' && !KNOWN_VERSIONS.has(version)) {
      throw new InvkFormatError('unsupported-version', `board.json version ${String(version)} is newer than this app`);
    }

    throw new InvkFormatError('damaged', `Invalid ${'board.json'}: ${parsed.error.issues[0]?.message ?? 'malformed'}`);
  }

  const boards: InvkBoard[] =
    parsed.data.version === 1
      ? [{ archived: false, isInbox: true, items: parsed.data.items, name: '' }]
      : parsed.data.boards;

  if (boards.filter((board) => board.isInbox).length !== 1) {
    throw new InvkFormatError('damaged', 'board.json must name exactly one inbox');
  }

  if (boards.reduce((count, board) => count + board.items.length, 0) > INVK_MAX_ENTRIES) {
    throw new InvkFormatError('damaged', 'board.json names more items than an archive can hold');
  }

  const seen = new Set<string>();

  for (const board of boards) {
    for (const item of board.items) {
      const key = `${item.kind}:${item.name}`;

      if (seen.has(key)) {
        throw new InvkFormatError('damaged', `Duplicate board entry ${key}`);
      }

      seen.add(key);
    }
  }

  return {
    boards: [...boards.filter((board) => board.isInbox), ...boards.filter((board) => !board.isInbox)].map((board) => ({
      ...board,
      items: [...board.items].sort(compareItems),
    })),
    version: 2,
  };
};

/** Build the entry from a server snapshot. Sorting here is what makes exports byte-comparable. */
export const buildInvkBoardSnapshot = (boards: readonly InvkBoard[]): InvkBoardSnapshot => {
  if (boards.length > INVK_MAX_BOARDS) {
    throw new InvkFormatError(
      'too-large',
      `Project has ${String(boards.length)} boards; an archive names at most ${String(INVK_MAX_BOARDS)}`
    );
  }

  return {
    boards: [...boards.filter((board) => board.isInbox), ...boards.filter((board) => !board.isInbox)].map((board) => ({
      archived: board.archived,
      isInbox: board.isInbox,
      items: [...board.items].sort(compareItems),
      name: board.name,
    })),
    version: 2,
  };
};
