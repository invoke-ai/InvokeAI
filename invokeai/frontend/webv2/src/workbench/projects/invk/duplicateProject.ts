import type { ProjectBoardSnapshotBoardDTO, ProjectRecordDTO } from '@workbench/projects/api';

import { type AccountScope, assertAccountScopeCurrent, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { createProjectSettled } from '@workbench/projects/api';
import { createProjectId } from '@workbench/projects/ids';
import {
  collectLiveAssetRefs,
  remapAssetRefs,
  selectCoverImageName,
  stripInstallationState,
} from '@workbench/projects/projectAssets';

import type { MediaMaterializer } from './restoreProjectMedia';
import type { ProjectTransferIssues } from './transfer';

import { type CopyMediaResult, copyImagesToBoard, copyVideosToBoard, isRequestCancellation } from './assetTransport';
import { INVK_MAX_BOARDS, type InvkBoard } from './board';
import { InvkFormatError, toInvkFormatReason } from './format';
import { createStagingBoards, findInboxStagingBoardId, placeMemberBoards } from './memberBoards';
import {
  createRestoredMediaLedger,
  restoreProjectMedia,
  rollbackRestoredMedia,
  rollbackUnlessProjectExists,
} from './restoreProjectMedia';
import { toMediaRefs } from './transfer';

/** Copy board media server-side and reuse external reference identities. */

export interface DuplicateProjectInput {
  /** The source project's boards with their visible contents, inbox first. */
  boards: readonly ProjectBoardSnapshotBoardDTO[];
  owner: AccountScope;
  /** The acknowledged source record — for an open project, flushed first. */
  record: ProjectRecordDTO;
  /** Stable identity reserved before the operation, used by retry-safe conflict copies. */
  identity?: { id: string; name: string };
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
 * Report each batch item's outcome and progress; cancellation aborts the operation rather than becoming a batch
 * failure.
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

/** Project creation commits the staging boards; precommit rollback deletes only ledger-owned resources. */
export const duplicateProjectRecord = async (
  input: DuplicateProjectInput,
  deps: DuplicateProjectDeps = {}
): Promise<DuplicateProjectResult> => {
  const { owner } = input;
  const id = input.identity?.id ?? createProjectId();
  const name = input.identity?.name ?? `${input.record.name} copy`;

  assertAccountScopeCurrent(owner);

  // Canonicalize with a fresh project identity and reset source board/selection state.
  const { deserializeProjectDocument } = await import('@workbench/projects/syncedPersistence');

  assertAccountScopeCurrent(owner);

  const loaded = deserializeProjectDocument({ ...stripInstallationState(input.record.data), id, name });

  if (loaded.status === 'refused') {
    throw new InvkFormatError(toInvkFormatReason(loaded.refused), 'The project document was refused.');
  }

  if (loaded.status !== 'loaded') {
    throw new InvkFormatError('damaged', 'The project document will not rehydrate.');
  }

  const project = loaded.project;

  const { applyAuthoritativeProjectBoard, serializeProjectDocument } =
    await import('@workbench/projects/projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const boards: InvkBoard[] = input.boards.map((board) => ({
    archived: board.archived,
    isInbox: board.is_inbox,
    items: board.items,
    name: board.name,
  }));

  // The same ceiling as an archive: a copy stages and moves each board one request at a time.
  if (boards.length > INVK_MAX_BOARDS) {
    throw new InvkFormatError(
      'too-large',
      `Project has ${String(boards.length)} boards; a copy carries at most ${String(INVK_MAX_BOARDS)}`
    );
  }
  // Made before anything is staged: every staging board joins it as it is created, so a copy abandoned at any point
  // can delete them all.
  const ledger = createRestoredMediaLedger([]);
  let didCreateProject = false;
  let didAttemptProjectCreate = false;

  try {
    assertAccountScopeCurrent(owner);

    const stagedBoards = await createStagingBoards(boards, name, ledger, owner.signal);
    const inboxStagingBoardId = findInboxStagingBoardId(stagedBoards);

    assertAccountScopeCurrent(owner);

    const restored = await restoreProjectMedia(
      {
        boards: stagedBoards.map(({ board, stagingBoardId }) => ({ items: board.items, stagingBoardId })),
        // Same-server duplication needs no bundled bytes and inherits unresolved external references.
        coverBytes: null,
        coverSourceImageName: selectCoverImageName(canonicalDocument),
        documentRefs: toMediaRefs(collectLiveAssetRefs(canonicalDocument)),
        ledger,
        projectId: id,
      },
      {
        // Reuse document-only names without existence probes; references may already be dangling.
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
    didAttemptProjectCreate = true;

    const record = await createProjectSettled(
      {
        data: remapAssetRefs(canonicalDocument, restored.mappings),
        minimum_canvas_schema_version: input.record.minimum_canvas_schema_version,
        name,
        project_id: id,
        ...(inboxStagingBoardId === null ? {} : { board_id: inboxStagingBoardId }),
      },
      owner
    );

    didCreateProject = true;
    assertAccountScopeCurrent(owner);
    // The copy exists; its other boards can now belong to it. Reported, never fatal, from here on.
    const boardIssues = await placeMemberBoards(stagedBoards, record.project_id, { signal: owner.signal });
    assertAccountScopeCurrent(owner);

    return {
      boardIssues,
      boardItemIssues: restored.boardItemIssues,
      coverImageName: restored.coverImageName,
      documentReferenceIssues: restored.documentReferenceIssues,
      record: {
        ...record,
        data: applyAuthoritativeProjectBoard(record.data, record.board_id, { selectBoard: true }),
      },
    };
  } catch (error) {
    const rollback = () => rollbackRestoredMedia(ledger, { signal: owner.signal });

    // Before the create was even attempted nothing is ambiguous: whatever was staged or copied is ours to drop.
    if (!didAttemptProjectCreate && isAccountScopeCurrent(owner)) {
      await rollback();
    } else {
      await rollbackUnlessProjectExists(error, didCreateProject, owner, rollback);
    }

    throw error;
  }
};
