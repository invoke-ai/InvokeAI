import type { ProjectBoardItemDTO, ProjectRecordDTO } from '@workbench/projects/api';

import { type AccountScope, assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { createProjectSettled } from '@workbench/projects/api';
import { createProjectId } from '@workbench/projects/ids';
import {
  collectLiveAssetRefs,
  remapAssetRefs,
  selectCoverImageName,
  stripInstallationState,
} from '@workbench/projects/projectAssets';

import type { InvkBoardItem } from './board';
import type { MediaMaterializer } from './restoreProjectMedia';
import type { ProjectTransferIssues } from './transfer';

import {
  type CopyMediaResult,
  copyImagesToBoard,
  copyVideosToBoard,
  createStagingBoard,
  isRequestCancellation,
} from './assetTransport';
import { InvkFormatError } from './format';
import {
  createRestoredMediaLedger,
  restoreProjectMedia,
  rollbackRestoredMedia,
  rollbackUnlessProjectExists,
} from './restoreProjectMedia';
import { toMediaRefs } from './transfer';

/**
 * Duplicating a project: the same restore as import, with the bytes taking a shortcut.
 *
 * A copy must own its media outright (see `transfer.ts`), which makes this import's problem. Only
 * materialization differs, and it differs because both projects live on this one server: the copy
 * endpoints work in place, so a 2 GiB board costs no traffic instead of 4 GiB and 2N requests.
 *
 * Document-only references already exist here and are not the project's own board content, so the
 * shared engine's existence check finds them all and nothing is uploaded.
 */

export interface DuplicateProjectInput {
  /** The board's visible contents, enumerated for the source project. */
  boardItems: readonly ProjectBoardItemDTO[];
  owner: AccountScope;
  /** The acknowledged source record — for an open project, flushed first. */
  record: ProjectRecordDTO;
}

export interface DuplicateProjectDeps {
  copyImages?: typeof copyImagesToBoard;
  copyVideos?: typeof copyVideosToBoard;
  onProgress?: (progress: { completed: number; total: number }) => void;
}

export interface DuplicateProjectResult extends ProjectTransferIssues {
  /** The cover to record for the copy, or `null` when the source had none this server can serve. */
  coverImageName: string | null;
  record: ProjectRecordDTO;
}

/**
 * The duplication half of the materialization seam: one bounded request sequence per kind, pixels
 * staying put. Progress moves per item once the sequence answers — there is no finer event.
 *
 * A batch that does not answer at all is every name in it failing, not the duplication failing:
 * the route reports per-item failures so one bad source cannot cost the batch, and letting a
 * transport error do what the route refuses would give that away at the last step.
 */
export const createCopyMediaMaterializer = (
  deps: { copyImages?: typeof copyImagesToBoard; copyVideos?: typeof copyVideosToBoard; signal?: AbortSignal } = {}
): MediaMaterializer => {
  const copyImages = deps.copyImages ?? copyImagesToBoard;
  const copyVideos = deps.copyVideos ?? copyVideosToBoard;
  const allFailed =
    (names: string[]) =>
    (error: unknown): CopyMediaResult => {
      if (isRequestCancellation(error)) {
        throw error;
      }

      return { copied: [], failed: names };
    };

  return async (items, boardId, onItemSettled) => {
    const imageNames = items.filter((item) => item.kind === 'image').map((item) => item.name);
    const videoNames = items.filter((item) => item.kind === 'video').map((item) => item.name);
    const [images, videos] = await Promise.all([
      copyImages(imageNames, boardId, deps.signal).catch(allFailed(imageNames)),
      copyVideos(videoNames, boardId, deps.signal).catch(allFailed(videoNames)),
    ]);

    for (let index = 0; index < items.length; index += 1) {
      onItemSettled();
    }

    return {
      failed: [
        ...images.failed.map((name) => ({ kind: 'image' as const, name, reason: 'upload-failed' as const })),
        ...videos.failed.map((name) => ({ kind: 'video' as const, name, reason: 'upload-failed' as const })),
      ],
      materialized: [
        ...images.copied.map((entry) => ({ kind: 'image' as const, name: entry.name, sourceName: entry.sourceName })),
        ...videos.copied.map((entry) => ({ kind: 'video' as const, name: entry.name, sourceName: entry.sourceName })),
      ],
    };
  };
};

/**
 * Copy a project and everything its board holds, under identities the copy owns.
 *
 * The commit point is the same as an import's: the media is materialized onto an unclaimed staging
 * board, and creating the project claims it. A failure before that deletes exactly what this
 * duplication made and leaves the original untouched.
 */
export const duplicateProjectRecord = async (
  input: DuplicateProjectInput,
  deps: DuplicateProjectDeps = {}
): Promise<DuplicateProjectResult> => {
  const { owner } = input;
  const id = createProjectId();
  const name = `${input.record.name} copy`;

  assertAccountScopeCurrent(owner);

  // Canonicalized through the reducer with the new identity, exactly as an import is: the copy is a
  // new project, not a second pointer at the original's board and gallery selection.
  const { deserializeProjectDocument } = await import('@workbench/projects/syncedPersistence');

  assertAccountScopeCurrent(owner);

  const project = deserializeProjectDocument({ ...stripInstallationState(input.record.data), id, name });

  if (!project) {
    throw new InvkFormatError('damaged', 'The project document will not rehydrate.');
  }

  const { applyAuthoritativeProjectBoard, serializeProjectDocument } =
    await import('@workbench/projects/projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const boardItems = input.boardItems as readonly InvkBoardItem[];
  const stagingBoardId = boardItems.length === 0 ? null : await createStagingBoard(name, owner.signal);
  const ledger = createRestoredMediaLedger(stagingBoardId);
  let didCreateProject = false;

  try {
    assertAccountScopeCurrent(owner);

    const restored = await restoreProjectMedia(
      {
        boardId: stagingBoardId,
        boardItems,
        // Nothing is bundled: every byte this project needs is already on this server, so a
        // reference the copy cannot resolve is one the original could not resolve either.
        coverBytes: null,
        coverSourceImageName: selectCoverImageName(canonicalDocument),
        documentRefs: toMediaRefs(collectLiveAssetRefs(canonicalDocument)),
        ledger,
        projectId: id,
      },
      {
        // Both projects live on this server, so every document-only reference resolves here by
        // definition — the check would ask the server whether it has media it just told us about.
        // For videos that is one request per referenced video, against a certain answer.
        findExistingImageNames: (names) => Promise.resolve(new Set(names)),
        findExistingVideoNames: (names) => Promise.resolve(new Set(names)),
        materializeBoardMedia: createCopyMediaMaterializer({
          ...(deps.copyImages === undefined ? {} : { copyImages: deps.copyImages }),
          ...(deps.copyVideos === undefined ? {} : { copyVideos: deps.copyVideos }),
          signal: owner.signal,
        }),
        ...(deps.onProgress === undefined ? {} : { onProgress: deps.onProgress }),
        signal: owner.signal,
      }
    );

    assertAccountScopeCurrent(owner);

    const record = await createProjectSettled(
      {
        data: remapAssetRefs(canonicalDocument, restored.mappings),
        name,
        project_id: id,
        ...(stagingBoardId === null ? {} : { board_id: stagingBoardId }),
      },
      owner
    );

    didCreateProject = true;
    assertAccountScopeCurrent(owner);

    return {
      boardItemIssues: restored.boardItemIssues,
      coverImageName: restored.coverImageName,
      documentReferenceIssues: restored.documentReferenceIssues,
      record: {
        ...record,
        data: applyAuthoritativeProjectBoard(record.data, record.board_id, { selectBoard: true }),
      },
    };
  } catch (error) {
    await rollbackUnlessProjectExists(error, didCreateProject, owner, () =>
      rollbackRestoredMedia(ledger, { signal: owner.signal })
    );

    throw error;
  }
};
