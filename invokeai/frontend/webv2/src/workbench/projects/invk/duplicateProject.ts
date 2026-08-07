import type { ProjectBoardItemDTO, ProjectRecordDTO } from '@workbench/projects/api';

import { type AccountScope, assertAccountScopeCurrent, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import {
  createProject as apiCreateProject,
  getProject as apiGetProject,
  isProjectNotFoundError,
} from '@workbench/projects/api';
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

import { copyImagesToBoard, copyVideosToBoard, createStagingBoard } from './assetTransport';
import { InvkFormatError } from './format';
import { createRestoredMediaLedger, restoreProjectMedia, rollbackRestoredMedia } from './restoreProjectMedia';
import { toMediaRefs } from './transfer';

/**
 * Duplicating a project: the same restore, with the bytes taking a shortcut.
 *
 * A copy has to own its media outright — `board_images` keys on the image name, so two projects
 * cannot share a board image — which makes duplication the same problem as import: materialize the
 * board under new identities, rewrite the document onto them, claim the staging board by creating
 * the project. Only the materialization differs, and it differs because both projects live on this
 * one server: the copy endpoints do it in place, so a 2 GiB board costs no traffic instead of 4 GiB
 * and 2N requests through a browser tab.
 *
 * Document-only references are the other half of that. They point at media that already exists here
 * and is not the project's own board content, so the copy reuses the identities rather than
 * duplicating them — the shared engine's existence check finds every one of them, and nothing is
 * uploaded.
 *
 * What the previous implementation did was `POST` the source's document with a new id and name. It
 * carried no media at all, so the copy's board was empty, and it did not canonicalize or strip
 * installation state either — the copy inherited the original's board id and gallery selection.
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
 * The duplication half of the materialization seam: one request per kind, and the pixels stay put.
 *
 * Progress moves per item once each batch answers rather than during it. The copy is a single
 * server-side request per kind, so there is no finer truth to report — and unlike an import, there
 * is no upload to sit and watch.
 */
export const createCopyMediaMaterializer = (
  deps: { copyImages?: typeof copyImagesToBoard; copyVideos?: typeof copyVideosToBoard; signal?: AbortSignal } = {}
): MediaMaterializer => {
  const copyImages = deps.copyImages ?? copyImagesToBoard;
  const copyVideos = deps.copyVideos ?? copyVideosToBoard;

  return async (items, boardId, onItemSettled) => {
    const imageNames = items.filter((item) => item.kind === 'image').map((item) => item.name);
    const videoNames = items.filter((item) => item.kind === 'video').map((item) => item.name);
    const [images, videos] = await Promise.all([
      copyImages(imageNames, boardId, deps.signal),
      copyVideos(videoNames, boardId, deps.signal),
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

/** A rejected create is safe to clean up only when its client-chosen id is confirmed absent. */
const isProjectConfirmedAbsent = async (projectId: string, owner: AccountScope): Promise<boolean> => {
  try {
    await apiGetProject(projectId, owner.signal);

    return false;
  } catch (error) {
    return isAccountScopeCurrent(owner) && isProjectNotFoundError(error);
  }
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

    const record = await apiCreateProject(
      {
        data: remapAssetRefs(canonicalDocument, restored.mappings),
        name,
        project_id: id,
        ...(stagingBoardId === null ? {} : { board_id: stagingBoardId }),
      },
      owner.signal
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
    const canRollback =
      !didCreateProject && isAccountScopeCurrent(owner) && (await isProjectConfirmedAbsent(id, owner));

    if (canRollback) {
      try {
        await rollbackRestoredMedia(ledger, { signal: owner.signal });
      } catch {
        // Best-effort: the create failure is the actionable result, and must never be replaced by
        // a failed delete request.
      }
    }

    throw error;
  }
};
