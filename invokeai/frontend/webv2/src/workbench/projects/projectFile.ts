import type { Project } from '@workbench/projectContracts';

import { downloadBlob } from '@platform/browser/downloadBlob';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import type { ProjectTransferIssues } from './invk/transfer';

import { createProjectSettled, getProjectBoardSnapshot, type ProjectRecordDTO } from './api';
import { recordProjectCover } from './covers';
import { createProjectId } from './ids';
import { INVK_EXTENSION, InvkFormatError } from './invk/format';
import { readAcknowledgedProject, upsertProjectSummary } from './library';
import { remapAssetRefs, stripInstallationState } from './projectAssets';

export const LEGACY_PROJECT_FILE_EXTENSION = '.invokeproject.json';

const PROJECT_FILE_KIND = 'invokeai-project';
const PROJECT_FILE_VERSION = 1;

/** The JSON envelope shipped before `.invk`. Read only — nothing writes one any more. */
interface ProjectFile {
  document: Record<string, unknown>;
  exportedAt: string;
  kind: typeof PROJECT_FILE_KIND;
  version: typeof PROJECT_FILE_VERSION;
}

/** Returns the embedded legacy document, or null when the text is not one of our exports. */
export const parseProjectFile = (text: string): Record<string, unknown> | null => {
  try {
    const parsed = JSON.parse(text) as Partial<ProjectFile> | null;

    if (
      !parsed ||
      parsed.kind !== PROJECT_FILE_KIND ||
      parsed.version !== PROJECT_FILE_VERSION ||
      !parsed.document ||
      typeof parsed.document !== 'object' ||
      Array.isArray(parsed.document)
    ) {
      return null;
    }

    return parsed.document as Record<string, unknown>;
  } catch {
    return null;
  }
};

/**
 * Export and import a project as an `.invk` archive. This module is the workflow; `./invk` is the
 * format.
 *
 * An imported document gets a fresh id, never the one in the file: two people exchanging a project
 * would otherwise collide the moment both saved.
 *
 * The ZIP codec and the workbench reducer are both `await import()`ed. The Launchpad offers Import
 * and Export on a route that never mounts the editor, and should not pay for either until someone
 * picks a file.
 */

/**
 * How far along a project file is.
 *
 * `bundling` and `restoring` count assets; `packing` is the single ZIP write at
 * the end, which has no unit worth counting and is reported so the caller can
 * stop showing a number that has stopped moving.
 */
export interface ProjectFileProgress {
  completed: number;
  phase: 'bundling' | 'packing' | 'restoring';
  total: number;
}

export interface ProjectFileOptions {
  onProgress?: (progress: ProjectFileProgress) => void;
  owner?: AccountScope;
}

export interface ProjectExportOutcome extends ProjectTransferIssues {
  /** The name the archive was downloaded under. */
  fileName: string;
}

export interface ProjectImportOutcome extends ProjectTransferIssues {
  record: ProjectRecordDTO;
}

const readProjectDocument = async (file: File) => {
  if (file.name.toLowerCase().endsWith(LEGACY_PROJECT_FILE_EXTENSION)) {
    const { INVK_MAX_ARCHIVE_BYTES } = await import('./invk/archive');

    if (file.size > INVK_MAX_ARCHIVE_BYTES) {
      throw new InvkFormatError('too-large', `Project file is ${file.size} bytes.`);
    }

    const projectDocument = parseProjectFile(await file.text());

    if (projectDocument === null) {
      throw new InvkFormatError('not-a-project', 'This JSON file is not an Invoke project export.');
    }

    return { format: 'legacy-json' as const, projectDocument };
  }

  const { readInvkArchive } = await import('./invk/importProject');

  return { contents: await readInvkArchive(file), format: 'invk' as const };
};

const exportProjectDocument = async (
  name: string,
  projectId: string,
  projectDocument: Record<string, unknown>,
  options: Required<Pick<ProjectFileOptions, 'owner'>> & ProjectFileOptions
): Promise<ProjectExportOutcome> => {
  const { executeInvkExport, planInvkExport } = await import('./invk/exportProject');
  const { onProgress, owner } = options;

  assertAccountScopeCurrent(owner);

  // Enumerated before anything is planned, and fatal if it fails. An archive whose `board.json`
  // silently said "empty" would be a lie the reader has no way to detect — worse than no archive.
  const snapshot = await getProjectBoardSnapshot(projectId, owner.signal);

  assertAccountScopeCurrent(owner);

  const plan = planInvkExport({
    appVersion: APP_VERSION,
    boardItems: snapshot.items,
    createdAt: new Date().toISOString(),
    name,
    projectDocument,
  });

  const result = await executeInvkExport(plan, {
    download: downloadBlob,
    signal: owner.signal,
    ...(onProgress === undefined ? {} : { onProgress }),
  });

  assertAccountScopeCurrent(owner);

  return {
    boardItemIssues: result.boardItemIssues,
    documentReferenceIssues: result.documentReferenceIssues,
    fileName: plan.fileName,
  };
};

/** Export a project from its server record, flushing it first — see {@link readAcknowledgedProject}. */
export const exportLibraryProject = async (
  projectId: string,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const record = await readAcknowledgedProject(projectId, owner);

  assertAccountScopeCurrent(owner);

  return exportProjectDocument(record.name, record.project_id, record.data, { ...options, owner });
};

/** Export an open project from its live in-memory document. */
export const exportOpenProject = async (
  project: Project,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const { serializeProjectDocument } = await import('./projectDocument');

  assertAccountScopeCurrent(owner);

  return exportProjectDocument(project.name, project.id, serializeProjectDocument(project), {
    ...options,
    owner,
  });
};

/**
 * Import an `.invk` as a new server project: restore its media, rewrite every reference the server
 * renamed, then create the project — claiming its staging board in the same request. Throws
 * {@link InvkFormatError} so callers can translate the reason.
 *
 * Creating the project is the commit point; see {@link createStagingBoard}. A v2 archive or a
 * legacy JSON document stages no board, and the server gives the new project an empty one.
 */
export const importProjectFile = async (
  file: File,
  options: ProjectFileOptions = {}
): Promise<ProjectImportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const source = await readProjectDocument(file);
  const projectDocument = source.format === 'invk' ? source.contents.projectDocument : source.projectDocument;

  assertAccountScopeCurrent(owner);

  const id = createProjectId();
  const name =
    typeof projectDocument.name === 'string' && projectDocument.name.trim()
      ? projectDocument.name.trim()
      : 'Imported project';
  // Stripped on the way in as well as on the way out, so the rule holds for documents this app did
  // not write: a legacy `.invokeproject.json`, an archive from a dev build, a hand-edited one. A
  // stranger's `selectedImage`/`compareImage` would otherwise arrive intact and unfixable — the
  // collector skips those keys, so the restore can neither fetch them nor report them as dangling.
  const candidate = { ...stripInstallationState(projectDocument), id, name };
  // Full validation rehydrates the document through the Workbench reducer, so
  // it is loaded here rather than imported: the Launchpad should not carry the
  // editor's aggregate state just to offer an Import button.
  const { deserializeProjectDocument } = await import('./syncedPersistence');

  assertAccountScopeCurrent(owner);

  const project = deserializeProjectDocument(candidate);

  if (!project) {
    throw new InvkFormatError('damaged', 'The project document will not rehydrate.');
  }

  const { applyAuthoritativeProjectBoard, serializeProjectDocument } = await import('./projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const archive = source.format === 'invk' ? source.contents : null;
  // Loaded only for an archive: a legacy JSON document restores nothing, so it has nothing to undo.
  const restoreMedia = archive === null ? null : await import('./invk/restoreProjectMedia');

  assertAccountScopeCurrent(owner);

  // Only an archive that actually carries board media needs somewhere to put it. An empty board is
  // the server's to create, and staging one would be an unclaimed board to leak for no gain.
  const stagingBoardId =
    archive?.boardSnapshot && archive.boardSnapshot.items.length > 0
      ? await (async () => {
          const { createStagingBoard } = await import('./invk/assetTransport');

          return createStagingBoard(name, owner.signal);
        })()
      : null;
  const ledger = restoreMedia?.createRestoredMediaLedger(stagingBoardId) ?? null;
  let didCreateProject = false;

  try {
    assertAccountScopeCurrent(owner);

    const restored =
      archive === null || ledger === null
        ? null
        : await (async () => {
            const { restoreArchiveMedia } = await import('./invk/importProject');

            return restoreArchiveMedia(
              archive,
              { boardId: stagingBoardId, ledger, projectDocument: canonicalDocument, projectId: id },
              {
                signal: owner.signal,
                ...(options.onProgress === undefined
                  ? {}
                  : {
                      onProgress: ({ completed, total }) =>
                        options.onProgress?.({ completed, phase: 'restoring', total }),
                    }),
              }
            );
          })();

    assertAccountScopeCurrent(owner);

    const document = restored === null ? canonicalDocument : remapAssetRefs(canonicalDocument, restored.mappings);
    const record = await createProjectSettled(
      {
        data: document,
        name,
        project_id: id,
        ...(stagingBoardId === null ? {} : { board_id: stagingBoardId }),
      },
      owner
    );

    didCreateProject = true;
    assertAccountScopeCurrent(owner);
    upsertProjectSummary({ id: record.project_id, name: record.name, revision: record.revision }, owner);

    if (restored?.coverImageName) {
      recordProjectCover(record.project_id, restored.coverImageName, owner);
    }

    return {
      boardItemIssues: restored?.boardItemIssues ?? [],
      documentReferenceIssues: restored?.documentReferenceIssues ?? [],
      // The board the server says it claimed, not the one we asked it to: a project meeting its
      // owner for the first time is also pointed at its board, which is what `selectBoard` means.
      record: {
        ...record,
        data: applyAuthoritativeProjectBoard(record.data, record.board_id, { selectBoard: true }),
      },
    };
  } catch (error) {
    // Reached through the lazily-loaded module, so a legacy JSON import still never pulls the
    // restore engine into the graph — it has no media to undo.
    if (ledger !== null && restoreMedia !== null) {
      await restoreMedia.rollbackUnlessProjectExists(error, didCreateProject, owner, () =>
        restoreMedia.rollbackRestoredMedia(ledger, { signal: owner.signal })
      );
    }

    throw error;
  }
};

/** Open the browser's file picker for a project file; null when dismissed. */
export const pickProjectFile = (owner: AccountScope = captureAccountScope()): Promise<File | null> =>
  new Promise((resolve) => {
    const input = document.createElement('input');
    let isSettled = false;

    const finish = (file: File | null): void => {
      if (isSettled) {
        return;
      }

      isSettled = true;
      owner.signal.removeEventListener('abort', handleAbort);
      input.onchange = null;
      input.oncancel = null;
      resolve(isAccountScopeCurrent(owner) ? file : null);
    };
    const handleAbort = (): void => finish(null);

    input.type = 'file';
    input.accept = `${INVK_EXTENSION},${LEGACY_PROJECT_FILE_EXTENSION}`;
    input.onchange = () => finish(input.files?.[0] ?? null);
    input.oncancel = () => finish(null);
    owner.signal.addEventListener('abort', handleAbort, { once: true });

    if (owner.signal.aborted) {
      finish(null);
      return;
    }

    input.click();
  });
