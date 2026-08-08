import { describe, expect, it } from 'vitest';

import { buildMissingMediaName, createTransferIssueLog, planMediaTransfer, toMediaKey, toMediaRefs } from './transfer';

const boardItem = (name: string, overrides: Record<string, unknown> = {}) =>
  ({ category: 'general', kind: 'image', name, starred: false, ...overrides }) as Parameters<
    typeof planMediaTransfer
  >[0][number];

describe('toMediaKey', () => {
  /** Images and videos are separate backend namespaces; one name can legitimately be both. */
  it('separates the two namespaces', () => {
    expect(toMediaKey({ kind: 'image', name: 'twin' })).not.toBe(toMediaKey({ kind: 'video', name: 'twin' }));
  });
});

describe('toMediaRefs', () => {
  it('flattens the collector per-kind sets into one sorted list', () => {
    expect(toMediaRefs({ images: new Set(['b.png', 'a.png']), videos: new Set(['c.mp4']) })).toEqual([
      { kind: 'image', name: 'a.png' },
      { kind: 'image', name: 'b.png' },
      { kind: 'video', name: 'c.mp4' },
    ]);
  });
});

describe('planMediaTransfer', () => {
  it('marks which side each item came from', () => {
    const plan = planMediaTransfer(
      [boardItem('board-only.png'), boardItem('both.png')],
      [
        { kind: 'image', name: 'both.png' },
        { kind: 'image', name: 'document-only.png' },
      ]
    );

    expect(
      plan.map(({ isBoardItem, isDocumentReference, name }) => ({ isBoardItem, isDocumentReference, name }))
    ).toEqual([
      { isBoardItem: true, isDocumentReference: false, name: 'board-only.png' },
      { isBoardItem: true, isDocumentReference: true, name: 'both.png' },
      { isBoardItem: false, isDocumentReference: true, name: 'document-only.png' },
    ]);
  });

  /** Fetching is the expensive half of an export; an overlap must not be paid for twice. */
  it('yields an overlapping item exactly once', () => {
    const plan = planMediaTransfer([boardItem('both.png')], [{ kind: 'image', name: 'both.png' }]);

    expect(plan).toHaveLength(1);
  });

  it('keeps the board descriptor only for board items', () => {
    const plan = planMediaTransfer([boardItem('a.png', { starred: true })], [{ kind: 'image', name: 'b.png' }]);

    expect(plan[0]!.boardItem).toMatchObject({ starred: true });
    expect(plan[1]!.boardItem).toBeNull();
  });

  it('does not merge an image and a video sharing a name', () => {
    const plan = planMediaTransfer([boardItem('twin')], [{ kind: 'video', name: 'twin' }]);

    expect(plan).toHaveLength(2);
    expect(plan.map((item) => item.kind)).toEqual(['image', 'video']);
  });

  it('is stable regardless of input order', () => {
    const refs = [
      { kind: 'image' as const, name: 'b.png' },
      { kind: 'image' as const, name: 'a.png' },
    ];

    expect(planMediaTransfer([boardItem('c.png')], refs).map((item) => item.name)).toEqual(['a.png', 'b.png', 'c.png']);
  });
});

describe('createTransferIssueLog', () => {
  /**
   * The same failure can cost a board result and a canvas layer at once. Reporting it against both
   * roles is not double-counting — it genuinely failed as both.
   */
  it('records an overlapping failure against both roles and sorts the result', () => {
    const log = createTransferIssueLog();

    log.addDocumentReferenceIssue({ kind: 'image', name: 'z.png' }, 'upload-failed');
    log.addBoardItemIssue({ kind: 'image', name: 'z.png' }, 'upload-failed');
    log.addBoardItemIssue({ kind: 'image', name: 'a.png' }, 'star-failed');

    expect(log.toIssues()).toEqual({
      boardItemIssues: [
        { kind: 'image', name: 'a.png', reason: 'star-failed' },
        { kind: 'image', name: 'z.png', reason: 'upload-failed' },
      ],
      documentReferenceIssues: [{ kind: 'image', name: 'z.png', reason: 'upload-failed' }],
    });
  });

  it('reports nothing for a clean transfer', () => {
    expect(createTransferIssueLog().toIssues()).toEqual({ boardItemIssues: [], documentReferenceIssues: [] });
  });
});

describe('buildMissingMediaName', () => {
  /**
   * A failed board copy must not leave the document pointing at the old name: the destination may
   * well have its own media under it — on the same server, during a duplication, it certainly does
   * — and the project would open showing a stranger's picture with nothing to indicate it is wrong.
   */
  it('is stable within one project and distinct across projects, kinds and items', () => {
    expect(buildMissingMediaName('p1', 'image', 0)).toBe(buildMissingMediaName('p1', 'image', 0));
    expect(buildMissingMediaName('p1', 'image', 0)).not.toBe(buildMissingMediaName('p2', 'image', 0));
    expect(buildMissingMediaName('p1', 'image', 0)).not.toBe(buildMissingMediaName('p1', 'video', 0));
    expect(buildMissingMediaName('p1', 'image', 0)).not.toBe(buildMissingMediaName('p1', 'image', 1));
  });

  it('is shaped like an ordinary media name so a lookup fails cleanly', () => {
    const name = buildMissingMediaName('project-abc', 'image', 3);

    expect(name).not.toContain('/');
    expect(name).not.toContain('\\');
    expect(name.length).toBeGreaterThan(0);
  });
});
