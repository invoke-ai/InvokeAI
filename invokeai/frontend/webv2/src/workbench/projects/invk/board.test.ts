import { describe, expect, it } from 'vitest';

import { buildInvkBoardSnapshot, parseInvkBoardSnapshot } from './board';
import { InvkFormatError } from './format';

/**
 * `board.json` is the only part of an archive that tells the reader what the project *had*, as
 * opposed to what it draws with. A reader that half-trusted it would restore the wrong thing
 * quietly, so everything here is a structural refusal rather than a warning.
 */

const item = (overrides: Partial<Parameters<typeof buildInvkBoardSnapshot>[0][number]> = {}) => ({
  category: 'general' as const,
  kind: 'image' as const,
  name: 'a.png',
  starred: false,
  ...overrides,
});

const expectRefusal = (data: unknown): void => {
  try {
    parseInvkBoardSnapshot(data);
    expect.unreachable('parseInvkBoardSnapshot should have thrown');
  } catch (error) {
    expect(error).toBeInstanceOf(InvkFormatError);
    expect((error as InvkFormatError).reason).toBe('damaged');
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

    expect(parseInvkBoardSnapshot({ items, version: 1 }).items).toHaveLength(5);
  });

  it('canonicalizes order so exports of the same board compare equal', () => {
    const shuffled = [item({ kind: 'video', name: 'z.mp4' }), item({ name: 'b.png' }), item({ name: 'a.png' })];

    expect(parseInvkBoardSnapshot({ items: shuffled, version: 1 }).items.map((entry) => entry.name)).toEqual([
      'a.png',
      'b.png',
      'z.mp4',
    ]);
  });

  it('treats an image and a video of the same name as different items', () => {
    const items = [item({ name: 'twin' }), item({ kind: 'video', name: 'twin' })];

    expect(parseInvkBoardSnapshot({ items, version: 1 }).items).toHaveLength(2);
  });

  it('accepts an empty board', () => {
    expect(parseInvkBoardSnapshot({ items: [], version: 1 })).toEqual({ items: [], version: 1 });
  });

  it('refuses a duplicate descriptor', () => {
    expectRefusal({ items: [item(), item()], version: 1 });
  });

  /** A name with a separator would either escape `images/` or fail to round-trip through the ZIP. */
  it.each(['dir/a.png', 'dir\\a.png', '..', '.', '', 'a\0b.png'])('refuses the unsafe name %o', (name) => {
    expectRefusal({ items: [item({ name })], version: 1 });
  });

  it('refuses a category the gallery does not show', () => {
    // `other` is the canvas's private category — it is never board membership.
    expectRefusal({ items: [item({ category: 'other' as never })], version: 1 });
  });

  it('refuses an unknown kind, a wrong version, and an unknown key', () => {
    expectRefusal({ items: [item({ kind: 'audio' as never })], version: 1 });
    expectRefusal({ items: [], version: 2 });
    expectRefusal({ extra: true, items: [], version: 1 });
  });

  it('refuses a descriptor missing its starred flag', () => {
    expectRefusal({ items: [{ category: 'general', kind: 'image', name: 'a.png' }], version: 1 });
  });
});

describe('buildInvkBoardSnapshot', () => {
  it('sorts and stamps the file version, without touching the input', () => {
    const items = [item({ name: 'b.png' }), item({ name: 'a.png' })];
    const snapshot = buildInvkBoardSnapshot(items);

    expect(snapshot).toEqual({
      items: [item({ name: 'a.png' }), item({ name: 'b.png' })],
      version: 1,
    });
    expect(items[0]!.name).toBe('b.png');
  });

  it('round-trips through the parser', () => {
    const snapshot = buildInvkBoardSnapshot([item({ kind: 'video', name: 'v.mp4', starred: true }), item()]);

    expect(parseInvkBoardSnapshot(JSON.parse(JSON.stringify(snapshot)))).toEqual(snapshot);
  });
});
