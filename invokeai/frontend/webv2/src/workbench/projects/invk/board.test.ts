import { describe, expect, it } from 'vitest';

import { buildInvkBoardSnapshot, flattenInvkBoardItems, parseInvkBoardSnapshot, type InvkBoardItem } from './board';
import { InvkFormatError } from './format';

/** Malformed board enumeration must fail before restore creates resources. */

const item = (overrides: Partial<InvkBoardItem> = {}): InvkBoardItem => ({
  category: 'general',
  kind: 'image',
  name: 'a.png',
  starred: false,
  ...overrides,
});

const inbox = (items: InvkBoardItem[], name = 'Project') => ({ archived: false, isInbox: true, items, name });
const member = (items: InvkBoardItem[], name = 'Member', archived = false) => ({
  archived,
  isInbox: false,
  items,
  name,
});

const expectRefusal = (data: unknown, reason = 'damaged'): void => {
  try {
    parseInvkBoardSnapshot(data);
    expect.unreachable('parseInvkBoardSnapshot should have thrown');
  } catch (error) {
    expect(error).toBeInstanceOf(InvkFormatError);
    expect((error as InvkFormatError).reason).toBe(reason);
  }
};

describe('parseInvkBoardSnapshot', () => {
  it('accepts every visible category of both kinds', () => {
    const items = [
      item({ category: 'general' }),
      item({ category: 'control', name: 'b.png' }),
      item({ category: 'mask', name: 'c.png' }),
      item({ category: 'user', name: 'd.png' }),
      item({ kind: 'video', name: 'e.mp4' }),
    ];

    expect(parseInvkBoardSnapshot({ boards: [inbox(items)], version: 2 }).boards[0]?.items).toHaveLength(5);
  });

  it('canonicalizes item order so exports of the same board compare equal', () => {
    const shuffled = [item({ kind: 'video', name: 'z.mp4' }), item({ name: 'b.png' }), item({ name: 'a.png' })];

    expect(
      parseInvkBoardSnapshot({ boards: [inbox(shuffled)], version: 2 }).boards[0]?.items.map((entry) => entry.name)
    ).toEqual(['a.png', 'b.png', 'z.mp4']);
  });

  it('puts the inbox first whatever order the file lists the boards in', () => {
    const parsed = parseInvkBoardSnapshot({
      boards: [member([item({ name: 'm.png' })], 'Site plan refs', true), inbox([item()])],
      version: 2,
    });

    expect(parsed.boards.map((board) => [board.name, board.isInbox, board.archived])).toEqual([
      ['Project', true, false],
      ['Site plan refs', false, true],
    ]);
    expect(flattenInvkBoardItems(parsed).map((entry) => entry.name)).toEqual(['a.png', 'm.png']);
  });

  /** Dev builds wrote version 1 with the inbox alone; it reads as a project with no other boards. */
  it('reads a version 1 entry as a lone inbox', () => {
    expect(parseInvkBoardSnapshot({ items: [item({ name: 'b.png' }), item()], version: 1 })).toEqual({
      boards: [{ archived: false, isInbox: true, items: [item(), item({ name: 'b.png' })], name: '' }],
      version: 2,
    });
  });

  it('treats an image and a video of the same name as different items', () => {
    const items = [item({ name: 'twin' }), item({ kind: 'video', name: 'twin' })];

    expect(parseInvkBoardSnapshot({ boards: [inbox(items)], version: 2 }).boards[0]?.items).toHaveLength(2);
  });

  it('accepts an empty inbox', () => {
    expect(parseInvkBoardSnapshot({ boards: [inbox([])], version: 2 })).toEqual({
      boards: [inbox([])],
      version: 2,
    });
  });

  it('refuses a duplicate descriptor, on one board or across two', () => {
    expectRefusal({ boards: [inbox([item(), item()])], version: 2 });
    expectRefusal({ boards: [inbox([item()]), member([item()])], version: 2 });
  });

  it('refuses a file without exactly one inbox', () => {
    expectRefusal({ boards: [member([])], version: 2 });
    expectRefusal({ boards: [inbox([]), inbox([])], version: 2 });
    expectRefusal({ boards: [], version: 2 });
  });

  /** The server allows a blank board name, so an export can carry one; refusing it would break the round-trip. */
  it('keeps a blank name on a board other than the inbox', () => {
    expect(parseInvkBoardSnapshot({ boards: [inbox([]), member([], '  ')], version: 2 }).boards[1]?.name).toBe('  ');
  });

  it('caps the items of all boards together, not each board alone', () => {
    const items = (count: number, prefix: string) =>
      Array.from({ length: count }, (_, index) => item({ name: `${prefix}${String(index)}.png` }));

    expect(
      parseInvkBoardSnapshot({ boards: [inbox(items(10_000, 'a')), member(items(10_000, 'b'))], version: 2 }).boards
    ).toHaveLength(2);
    expectRefusal({ boards: [inbox(items(10_001, 'a')), member(items(10_000, 'b'))], version: 2 });
  });

  /** A name with a separator would either escape `images/` or fail to round-trip through the ZIP. */
  it.each(['dir/a.png', 'dir\\a.png', '..', '.', '', 'a\0b.png'])('refuses the unsafe name %o', (name) => {
    expectRefusal({ boards: [inbox([item({ name })])], version: 2 });
  });

  it('refuses a category the gallery does not show', () => {
    // `other` is the canvas's private category — it is never board membership.
    expectRefusal({ boards: [inbox([item({ category: 'other' as never })])], version: 2 });
  });

  it('refuses an unknown kind, a malformed known version, and an unknown key', () => {
    expectRefusal({ boards: [inbox([item({ kind: 'audio' as never })])], version: 2 });
    expectRefusal({ items: [], version: 2 });
    expectRefusal({ boards: [inbox([])], extra: true, version: 2 });
    expectRefusal({ boards: [{ ...inbox([]), extra: true }], version: 2 });
  });

  it('tells a newer version apart from a damaged file', () => {
    expectRefusal({ boards: [inbox([])], version: 3 }, 'unsupported-version');
    expectRefusal({ version: 'two' });
  });

  it('refuses a descriptor missing its starred flag', () => {
    expectRefusal({
      boards: [
        { archived: false, isInbox: true, items: [{ category: 'general', kind: 'image', name: 'a.png' }], name: 'P' },
      ],
      version: 2,
    });
  });
});

describe('buildInvkBoardSnapshot', () => {
  it('sorts items, leads with the inbox and stamps the file version, without touching the input', () => {
    const items = [item({ name: 'b.png' }), item({ name: 'a.png' })];
    const snapshot = buildInvkBoardSnapshot([member([item({ name: 'm.png' })]), inbox(items)]);

    expect(snapshot).toEqual({
      boards: [inbox([item({ name: 'a.png' }), item({ name: 'b.png' })]), member([item({ name: 'm.png' })])],
      version: 2,
    });
    expect(items[0]!.name).toBe('b.png');
  });

  it('round-trips through the parser', () => {
    const snapshot = buildInvkBoardSnapshot([
      inbox([item({ kind: 'video', name: 'v.mp4', starred: true }), item()]),
      member([item({ name: 'm.png' })], 'Old', true),
    ]);

    expect(parseInvkBoardSnapshot(JSON.parse(JSON.stringify(snapshot)))).toEqual(snapshot);
  });
});
