import { describe, expect, it, vi } from 'vitest';

import type { InvkBoard } from './board';

import { createStagingBoards, findInboxStagingBoardId, placeMemberBoards } from './memberBoards';
import { createRestoredMediaLedger } from './restoreProjectMedia';

const item = { category: 'general' as const, kind: 'image' as const, name: 'a.png', starred: false };
const inbox: InvkBoard = { archived: false, isInbox: true, items: [item], name: 'Stored name' };
const old: InvkBoard = { archived: true, isInbox: false, items: [{ ...item, name: 'old.png' }], name: 'Old' };
const empty: InvkBoard = { archived: false, isInbox: false, items: [], name: 'Empty' };

describe('createStagingBoards', () => {
  it('stages every board in order, naming the inbox after the project, and finds the inbox among them', async () => {
    const ledger = createRestoredMediaLedger([]);
    const create = vi.fn((name: string) => Promise.resolve(`staged:${name}`));

    const staged = await createStagingBoards([inbox, old, empty], 'Project', ledger, undefined, create);

    expect(staged.map(({ stagingBoardId }) => stagingBoardId)).toEqual([
      'staged:Project',
      'staged:Old',
      'staged:Empty',
    ]);
    expect(staged.map(({ board }) => board)).toEqual([inbox, old, empty]);
    expect(findInboxStagingBoardId(staged)).toBe('staged:Project');
    expect(findInboxStagingBoardId([])).toBeNull();
  });

  it('puts each staging board on the ledger as it is made, so a failed later create leaves none unowned', async () => {
    const ledger = createRestoredMediaLedger([]);
    const create = vi
      .fn<(name: string) => Promise<string>>()
      .mockResolvedValueOnce('staged-inbox')
      .mockRejectedValueOnce(new Error('refused'));

    await expect(createStagingBoards([inbox, old], 'Project', ledger, undefined, create)).rejects.toThrow('refused');
    expect(ledger.boardIds).toEqual(['staged-inbox']);
  });
});

describe('placeMemberBoards', () => {
  const staged = [
    { board: inbox, stagingBoardId: 'staged-inbox' },
    { board: old, stagingBoardId: 'staged-old' },
    { board: empty, stagingBoardId: 'staged-empty' },
  ];
  const deps = () => ({ placeBoardInProject: vi.fn(() => Promise.resolve()) });

  it('moves every board but the inbox into the project as it was', async () => {
    const d = deps();

    await expect(placeMemberBoards(staged, 'p1', d)).resolves.toEqual([]);
    expect(d.placeBoardInProject.mock.calls).toEqual([
      ['staged-old', 'p1', true, undefined],
      ['staged-empty', 'p1', false, undefined],
    ]);
  });

  it('reports each board that could not move and carries on with the rest', async () => {
    const d = deps();
    d.placeBoardInProject.mockRejectedValueOnce(new Error('refused'));

    await expect(placeMemberBoards(staged, 'p1', d)).resolves.toEqual([{ name: 'Old' }]);
    expect(d.placeBoardInProject).toHaveBeenCalledTimes(2);
  });

  it('ends the placement when the operation was cancelled', async () => {
    const d = deps();
    d.placeBoardInProject.mockRejectedValueOnce(new DOMException('gone', 'AbortError'));

    await expect(placeMemberBoards(staged, 'p1', d)).rejects.toThrow('gone');
    expect(d.placeBoardInProject).toHaveBeenCalledTimes(1);
  });
});
