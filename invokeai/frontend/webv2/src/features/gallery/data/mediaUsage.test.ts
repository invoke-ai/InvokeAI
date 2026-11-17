import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetchJson: mocks.apiFetchJson,
}));

import { checkGalleryMediaUsage } from './mediaUsage';

describe('checkGalleryMediaUsage', () => {
  const usageBody = (overrides: Record<string, unknown> = {}) => ({
    checked_image_count: 2,
    checked_video_count: 1,
    documents: [
      {
        id: 'legacy_editor',
        image_count: 1,
        image_names: ['a.png'],
        kind: 'legacy_editor',
        name: null,
        revision: null,
        video_count: 0,
        video_names: [],
        workspace_revision: null,
      },
      {
        id: 'project-1',
        image_count: 1200,
        image_names: ['a.png', 'b.png'],
        kind: 'project',
        name: 'Harbor Tower',
        revision: 7,
        video_count: 1,
        video_names: ['c.mp4'],
        workspace_revision: 3,
      },
      {
        id: 'later-kind',
        image_count: 1,
        image_names: ['b.png'],
        kind: 'storyboard',
        name: '  ',
        // Revisions are only a hint: a malformed one reads as unknown rather than failing the check.
        revision: '2',
        video_count: 0,
        video_names: [],
        workspace_revision: -1,
      },
    ],
    documents_total: 140,
    in_use_image_count: 2,
    in_use_video_count: 1,
    other_accounts: { document_count: 3, image_count: 1, queue_item_count: 2, video_count: 0 },
    queue: { image_count: 1, image_names: ['a.png'], item_count: 2, video_count: 0, video_names: [] },
    ...overrides,
  });

  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
  });

  it('asks about the items by kind, or about a whole board, and maps the answer', async () => {
    mocks.apiFetchJson.mockResolvedValue(usageBody());

    const usage = await checkGalleryMediaUsage({
      kind: 'items',
      refs: [
        { kind: 'image', name: 'a.png' },
        { kind: 'video', name: 'c.mp4' },
        { kind: 'image', name: 'b.png' },
      ],
    });
    await checkGalleryMediaUsage({ boardId: 'board-1', kind: 'board' });

    expect(JSON.parse(String(mocks.apiFetchJson.mock.calls[0]?.[1]?.body))).toEqual({
      image_names: ['a.png', 'b.png'],
      video_names: ['c.mp4'],
    });
    expect(mocks.apiFetchJson.mock.calls[0]?.[0]).toBe('/api/v1/gallery/usage');
    expect(mocks.apiFetchJson.mock.calls[0]?.[1]?.method).toBe('POST');
    expect(JSON.parse(String(mocks.apiFetchJson.mock.calls[1]?.[1]?.body))).toEqual({ board_id: 'board-1' });
    expect(usage).toEqual({
      checkedImageCount: 2,
      checkedVideoCount: 1,
      documents: [
        {
          id: 'legacy_editor',
          imageCount: 1,
          imageNames: ['a.png'],
          kind: 'legacyEditor',
          name: null,
          revision: null,
          videoCount: 0,
          videoNames: [],
          workspaceRevision: null,
        },
        {
          id: 'project-1',
          imageCount: 1200,
          imageNames: ['a.png', 'b.png'],
          kind: 'project',
          name: 'Harbor Tower',
          revision: 7,
          videoCount: 1,
          videoNames: ['c.mp4'],
          workspaceRevision: 3,
        },
        // A kind this build does not know still uses the media; a blank name reads as none.
        {
          id: 'later-kind',
          imageCount: 1,
          imageNames: ['b.png'],
          kind: 'other',
          name: null,
          revision: null,
          videoCount: 0,
          videoNames: [],
          workspaceRevision: null,
        },
      ],
      documentsTotal: 140,
      inUseImageCount: 2,
      inUseVideoCount: 1,
      otherAccounts: { documentCount: 3, imageCount: 1, queueItemCount: 2, videoCount: 0 },
      queue: { imageCount: 1, imageNames: ['a.png'], itemCount: 2, videoCount: 0, videoNames: [] },
    });
  });

  it('fails on a malformed answer rather than reading it as nothing in use', async () => {
    mocks.apiFetchJson.mockResolvedValue(usageBody({ in_use_image_count: undefined }));

    await expect(checkGalleryMediaUsage({ boardId: 'board-1', kind: 'board' })).rejects.toThrow(TypeError);
  });

  it('drops an answer that arrives after the account changed', async () => {
    let answer: (value: unknown) => void = () => {};
    mocks.apiFetchJson.mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      })
    );

    const check = checkGalleryMediaUsage({ boardId: 'board-1', kind: 'board' });
    const signal = mocks.apiFetchJson.mock.calls[0]?.[1]?.signal as AbortSignal;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    answer(usageBody());

    await expect(check).rejects.toThrow('no longer active');
    expect(signal.aborted).toBe(true);
  });

  describe('past one request’s cap', () => {
    const names = (prefix: string, count: number) => Array.from({ length: count }, (_, index) => `${prefix}-${index}`);
    const document = (kind: string, id: string, imageNames: string[], revision: number | null = null) => ({
      id,
      image_count: imageNames.length,
      image_names: imageNames,
      kind,
      name: id,
      revision,
      video_count: 0,
      video_names: [],
      workspace_revision: revision === null ? null : 3,
    });
    const refs = [
      ...names('img', 1500).map((name) => ({ kind: 'image' as const, name })),
      ...names('clip', 200).map((name) => ({ kind: 'video' as const, name })),
    ];

    it('asks in batches of at most the cap per kind, one after another, and merges the answers', async () => {
      mocks.apiFetchJson
        .mockResolvedValueOnce(
          usageBody({
            checked_image_count: 1000,
            checked_video_count: 200,
            // A third document past the listing cap.
            documents: [document('workflow', 'wf-1', ['img-2']), document('project', 'project-1', ['img-1'], 7)],
            documents_total: 3,
            in_use_image_count: 2,
            in_use_video_count: 0,
            other_accounts: { document_count: 1, image_count: 1, queue_item_count: 0, video_count: 0 },
            queue: { image_count: 1, image_names: ['img-3'], item_count: 2, video_count: 0, video_names: [] },
          })
        )
        .mockResolvedValueOnce(
          usageBody({
            checked_image_count: 500,
            checked_video_count: 0,
            // Saved between the two requests: its revisions no longer describe all of its names.
            documents: [document('project', 'project-1', ['img-1200'], 8)],
            documents_total: 1,
            in_use_image_count: 1,
            in_use_video_count: 0,
            other_accounts: { document_count: 1, image_count: 1, queue_item_count: 1, video_count: 0 },
            queue: { image_count: 1, image_names: ['img-1100'], item_count: 1, video_count: 0, video_names: [] },
          })
        );

      const usage = await checkGalleryMediaUsage({ kind: 'items', refs });

      expect(mocks.apiFetchJson.mock.calls.map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { image_names: names('img', 1500).slice(0, 1000), video_names: names('clip', 200) },
        { image_names: names('img', 1500).slice(1000), video_names: [] },
      ]);
      expect(usage).toEqual({
        checkedImageCount: 1500,
        checkedVideoCount: 200,
        documents: [
          {
            id: 'project-1',
            imageCount: 2,
            imageNames: ['img-1', 'img-1200'],
            kind: 'project',
            name: 'project-1',
            revision: null,
            videoCount: 0,
            videoNames: [],
            workspaceRevision: null,
          },
          {
            id: 'wf-1',
            imageCount: 1,
            imageNames: ['img-2'],
            kind: 'workflow',
            name: 'wf-1',
            revision: null,
            videoCount: 0,
            videoNames: [],
            workspaceRevision: null,
          },
        ],
        documentsTotal: 3,
        inUseImageCount: 3,
        inUseVideoCount: 0,
        // Jobs and other accounts' documents are counted once per batch that found them: never fewer than there are.
        otherAccounts: { documentCount: 2, imageCount: 2, queueItemCount: 1, videoCount: 0 },
        queue: { imageCount: 2, imageNames: ['img-3', 'img-1100'], itemCount: 3, videoCount: 0, videoNames: [] },
      });
    });

    it('marks a document one batch left out past its listing cap as naming only some of its uses', async () => {
      const images = names('img', 2000).map((name) => ({ kind: 'image' as const, name }));
      // Project X uses 500 images of the first batch, but that batch names only the first 100 of its 150 documents.
      const listedFirst = Array.from({ length: 100 }, (_, index) => document('workflow', `wf-${index}`, ['img-1']));
      mocks.apiFetchJson
        .mockResolvedValueOnce(usageBody({ documents: listedFirst, documents_total: 150 }))
        .mockResolvedValueOnce(
          usageBody({
            documents: [
              document('project', 'project-x', ['img-1001', 'img-1002', 'img-1003'], 9),
              // Only this batch found it, and no batch hid anything from it but the first, which lists it.
              document('workflow', 'wf-0', ['img-1500']),
            ],
            documents_total: 2,
          })
        );

      const usage = await checkGalleryMediaUsage({ kind: 'items', refs: images });

      expect(usage.documents.find((entry) => entry.id === 'project-x')).toEqual({
        id: 'project-x',
        imageCount: 3,
        imageNames: ['img-1001', 'img-1002', 'img-1003'],
        kind: 'project',
        name: 'project-x',
        namesIncomplete: true,
        revision: null,
        videoCount: 0,
        videoNames: [],
        workspaceRevision: null,
      });
      expect(usage.documents.find((entry) => entry.id === 'wf-0')).toMatchObject({
        imageCount: 2,
        imageNames: ['img-1', 'img-1500'],
      });
      expect(usage.documents.find((entry) => entry.id === 'wf-0')).not.toHaveProperty('namesIncomplete');
      // The first batch's 50 unlisted documents are counted as further documents.
      expect(usage.documentsTotal).toBe(101 + 50);
    });

    it('keeps revisions every batch agrees on, and a document only one batch found as it was found', async () => {
      mocks.apiFetchJson
        .mockResolvedValueOnce(
          usageBody({ documents: [document('project', 'same', ['img-1'], 7)], documents_total: 1 })
        )
        .mockResolvedValueOnce(
          usageBody({
            documents: [document('project', 'same', ['img-1001'], 7), document('project', 'late', ['img-1002'], 4)],
            documents_total: 2,
          })
        );

      const usage = await checkGalleryMediaUsage({ kind: 'items', refs });

      expect(
        usage.documents.map(({ id, imageNames, revision, workspaceRevision }) => ({
          id,
          imageNames,
          revision,
          workspaceRevision,
        }))
      ).toEqual([
        { id: 'late', imageNames: ['img-1002'], revision: 4, workspaceRevision: 3 },
        { id: 'same', imageNames: ['img-1', 'img-1001'], revision: 7, workspaceRevision: 3 },
      ]);
    });

    it('keeps documents of different unknown kinds with the same id apart', async () => {
      mocks.apiFetchJson
        .mockResolvedValueOnce(
          usageBody({ documents: [document('storyboard', 'shared-id', ['img-1'])], documents_total: 1 })
        )
        .mockResolvedValueOnce(
          usageBody({ documents: [document('moodboard', 'shared-id', ['img-1001'])], documents_total: 1 })
        );

      const usage = await checkGalleryMediaUsage({ kind: 'items', refs });

      expect(usage.documents.map(({ imageNames, kind }) => ({ imageNames, kind }))).toEqual([
        { imageNames: ['img-1'], kind: 'other' },
        { imageNames: ['img-1001'], kind: 'other' },
      ]);
    });

    it('fails the whole check when a later batch fails', async () => {
      mocks.apiFetchJson
        .mockResolvedValueOnce(usageBody())
        .mockRejectedValueOnce(new ApiError('{"detail":"Server error"}', 500));

      await expect(checkGalleryMediaUsage({ kind: 'items', refs })).rejects.toBeInstanceOf(ApiError);
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    });

    it('asks no further once the caller aborts between batches', async () => {
      const controller = new AbortController();
      mocks.apiFetchJson.mockImplementationOnce(() => {
        controller.abort();
        return Promise.resolve(usageBody());
      });

      await expect(checkGalleryMediaUsage({ kind: 'items', refs }, controller.signal)).rejects.toMatchObject({
        name: 'AbortError',
      });
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    });

    it('asks no further once the account changes between batches', async () => {
      mocks.apiFetchJson.mockImplementationOnce(() => {
        accountLifecycle.invalidate();
        accountLifecycle.activate('user-b');
        return Promise.resolve(usageBody());
      });

      await expect(checkGalleryMediaUsage({ kind: 'items', refs })).rejects.toThrow('no longer active');
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
    });
  });
});
