import type { GalleryImage } from '@features/gallery';
import type { Project } from '@workbench/projectContracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { retryStagedResultBoard, saveStagedResultToGallery } from './saveStagedResultToGallery';

/** A project whose Gallery follows `selectedBoardId`, which is then where Canvas saves land. */
const createProject = (selectedBoardId: string): Project => {
  const project = structuredClone(createInitialWorkbenchState().projects[0]!);
  const gallery = getProjectWidgetInstance(project, 'gallery')!;

  gallery.state.values = { ...gallery.state.values, selectedBoardId };

  return project;
};

const promoted = (imageName: string, boardId = 'none'): GalleryImage => ({
  boardId,
  height: 512,
  imageCategory: 'general',
  imageName,
  imageUrl: `/full/${imageName}`,
  queuedAt: '2026-10-05T00:00:00.000Z',
  sourceQueueItemId: 'queue-item',
  starred: false,
  thumbnailUrl: `/thumb/${imageName}`,
  width: 512,
});

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

beforeEach(() => {
  accountLifecycle.activate('staged-save-user');
});

describe('saveStagedResultToGallery', () => {
  it('puts an unassigned result on the board Canvas saves go to', async () => {
    const addToBoard = vi.fn((_boardId: string, names: string[]) => Promise.resolve(names));

    const outcome = await saveStagedResultToGallery(
      { imageName: 'staged.png', project: createProject('board-a') },
      { addToBoard, promote: (name) => Promise.resolve(promoted(name)) }
    );

    expect(outcome).toEqual({ boardId: 'board-a', imageName: 'staged.png', status: 'saved' });
    expect(addToBoard).toHaveBeenCalledExactlyOnceWith('board-a', ['staged.png'], expect.any(AbortSignal));
  });

  it('leaves a result in Uncategorized when Canvas saves go there, without a board request', async () => {
    const addToBoard = vi.fn();

    const outcome = await saveStagedResultToGallery(
      { imageName: 'staged.png', project: createProject('none') },
      { addToBoard, promote: (name) => Promise.resolve(promoted(name)) }
    );

    expect(outcome).toEqual({ boardId: 'none', imageName: 'staged.png', status: 'saved' });
    expect(addToBoard).not.toHaveBeenCalled();
  });

  it('keeps a board the result is already on, so a repeated save re-confirms it rather than moving it', async () => {
    const addToBoard = vi.fn();

    const outcome = await saveStagedResultToGallery(
      { imageName: 'staged.png', project: createProject('board-a') },
      { addToBoard, promote: (name) => Promise.resolve(promoted(name, 'board-b')) }
    );

    expect(outcome).toEqual({ boardId: 'board-b', imageName: 'staged.png', status: 'saved' });
    expect(addToBoard).not.toHaveBeenCalled();
  });

  it('reads the destination before promoting, so a later change to the project cannot redirect it', async () => {
    const project = createProject('board-a');
    const promotion = deferred<GalleryImage>();
    const addToBoard = vi.fn((_boardId: string, names: string[]) => Promise.resolve(names));
    const saving = saveStagedResultToGallery(
      { imageName: 'staged.png', project },
      { addToBoard, promote: () => promotion.promise }
    );

    // What a project switch or a new destination choice does to the captured snapshot's successor.
    getProjectWidgetInstance(project, 'gallery')!.state.values = { selectedBoardId: 'board-z' };
    promotion.resolve(promoted('staged.png'));

    expect(await saving).toMatchObject({ boardId: 'board-a', status: 'saved' });
    expect(addToBoard).toHaveBeenCalledWith('board-a', ['staged.png'], expect.any(AbortSignal));
  });

  it.each([
    ['rejects', () => Promise.reject(new Error('board gone'))],
    ['skips the image', () => Promise.resolve([])],
  ])('reports the board alone as failed when the assignment %s', async (_case, addToBoard) => {
    const outcome = await saveStagedResultToGallery(
      { imageName: 'staged.png', project: createProject('board-a') },
      { addToBoard, promote: (name) => Promise.resolve(promoted(name)) }
    );

    expect(outcome).toEqual({ boardId: 'board-a', imageName: 'staged.png', status: 'board-failed' });
  });

  it('throws a failed promotion and assigns nothing', async () => {
    const addToBoard = vi.fn();

    await expect(
      saveStagedResultToGallery(
        { imageName: 'staged.png', project: createProject('board-a') },
        { addToBoard, promote: () => Promise.reject(new Error('offline')) }
      )
    ).rejects.toThrow('offline');
    expect(addToBoard).not.toHaveBeenCalled();
  });

  it('settles stale, assigning nothing, when the account changes during promotion', async () => {
    const promotion = deferred<GalleryImage>();
    const addToBoard = vi.fn();
    const saving = saveStagedResultToGallery(
      { imageName: 'staged.png', project: createProject('board-a') },
      { addToBoard, promote: () => promotion.promise }
    );

    accountLifecycle.activate('another-user');
    promotion.resolve(promoted('staged.png'));

    expect(await saving).toEqual({ status: 'stale' });
    expect(addToBoard).not.toHaveBeenCalled();
  });
});

describe('retryStagedResultBoard', () => {
  it('asks for the board again', async () => {
    const addToBoard = vi.fn((_boardId: string, names: string[]) => Promise.resolve(names));

    expect(await retryStagedResultBoard({ boardId: 'board-a', imageName: 'staged.png' }, { addToBoard })).toEqual({
      boardId: 'board-a',
      imageName: 'staged.png',
      status: 'saved',
    });
  });
});
