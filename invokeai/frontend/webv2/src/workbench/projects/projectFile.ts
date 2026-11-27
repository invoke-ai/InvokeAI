import type { Project } from '@workbench/projectContracts';

import { downloadBlob } from '@platform/browser/downloadBlob';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getProjectCanvasSchemaRequirement } from '@workbench/canvasSchemaVersion';
import { isViewerAutoSwitchMode } from '@workbench/settings/projectSettings';

import type { ProjectTransferIssues } from './invk/transfer';
import type { ProjectDocumentV3 } from './projectDocument';

import {
  createProjectSettled,
  getProjectBoardSnapshot,
  invalidateBoardLists,
  type ProjectWithWorkspaceDTO,
} from './api';
import { recordProjectCover } from './covers';
import { PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';
import { createProjectId } from './ids';
import { INVK_EXTENSION, InvkFormatError, toInvkRefusalError } from './invk/format';
import { readAcknowledgedProject, upsertProjectSummary } from './library';
import { remapAssetRefs, stripInstallationState } from './projectAssets';
import { ProjectWorkspaceInseparableError } from './projectFileErrors';
import { splitProjectDocument } from './projectHalves';
import { getOpenProject } from './syncStore';

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

/** Asset-count phases exclude ZIP packing. */
export interface ProjectFileProgress {
  completed: number;
  phase: 'bundling' | 'packing' | 'placing-boards' | 'restoring' | 'restoring-fonts';
  total: number;
}

export interface ProjectFileOptions {
  onProgress?: (progress: ProjectFileProgress) => void;
  owner?: AccountScope;
  includeFonts?: boolean;
  includeWorkspace?: boolean;
  skipEmbeddedFonts?: boolean;
}

interface ArchivedProject {
  content: Record<string, unknown>;
  documentSchemaVersion: number;
  workspace: Record<string, unknown> | null;
}

export interface ProjectExportOutcome extends ProjectTransferIssues {
  fileName: string;
}

export interface ProjectImportOutcome extends ProjectTransferIssues {
  record: ProjectWithWorkspaceDTO;
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

const toArchivedProject = (document: ProjectDocumentV3, includeWorkspace: boolean): ArchivedProject => {
  const halves = splitProjectDocument(document);

  return {
    content: halves.content as unknown as Record<string, unknown>,
    documentSchemaVersion: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    workspace: includeWorkspace ? (halves.workspace as unknown as Record<string, unknown>) : null,
  };
};

/** Verbatim, so a project this build cannot read can still be recovered. */
const toRecoveryArchive = (record: ProjectWithWorkspaceDTO, includeWorkspace: boolean): ArchivedProject => {
  const { documentSchemaVersion } = record.data;
  const schema = typeof documentSchemaVersion === 'number' ? documentSchemaVersion : 1;
  const isSplit = schema >= PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION;

  if (!isSplit && !includeWorkspace) {
    throw new ProjectWorkspaceInseparableError();
  }

  return {
    content: record.data,
    documentSchemaVersion: schema,
    workspace: isSplit && includeWorkspace ? (record.workspace?.data ?? null) : null,
  };
};

const exportProjectDocument = async (
  name: string,
  projectId: string,
  archived: ArchivedProject,
  minimumCanvasSchemaVersion: number,
  options: Required<Pick<ProjectFileOptions, 'owner'>> & ProjectFileOptions
): Promise<ProjectExportOutcome> => {
  const { executeInvkExport, planInvkExport } = await import('./invk/exportProject');
  const { onProgress, owner } = options;

  assertAccountScopeCurrent(owner);

  // Board enumeration failure must abort export rather than describe a falsely empty board.
  const snapshot = await getProjectBoardSnapshot(projectId, owner.signal);

  assertAccountScopeCurrent(owner);

  const plan = planInvkExport({
    appVersion: APP_VERSION,
    boards: snapshot.boards.map((board) => ({
      archived: board.archived,
      isInbox: board.is_inbox,
      items: board.items,
      name: board.name,
    })),
    content: archived.content,
    createdAt: new Date().toISOString(),
    documentSchemaVersion: archived.documentSchemaVersion,
    minimumCanvasSchemaVersion,
    name,
    workspace: archived.workspace,
    includeFonts: options.includeFonts ?? false,
  });

  const result = await executeInvkExport(plan, {
    download: downloadBlob,
    signal: owner.signal,
    ...(onProgress === undefined ? {} : { onProgress }),
  });

  assertAccountScopeCurrent(owner);

  return {
    boardIssues: result.boardIssues,
    boardItemIssues: result.boardItemIssues,
    documentReferenceIssues: result.documentReferenceIssues,
    fileName: plan.fileName,
  };
};

export const exportLibraryProject = async (
  projectId: string,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const record = await readAcknowledgedProject(projectId, owner);

  assertAccountScopeCurrent(owner);

  const [{ deserializeProjectRecord, isSplitProjectRecord }, { serializeProjectDocumentV3 }] = await Promise.all([
    import('./projectHydration'),
    import('./projectDocument'),
  ]);
  const loaded = deserializeProjectRecord(record);
  const includeWorkspace = options.includeWorkspace ?? true;
  const archived =
    loaded.status === 'loaded'
      ? // Of a split project the caller has never worked in, the workspace is defaults: nothing of theirs to carry.
        toArchivedProject(
          serializeProjectDocumentV3(loaded.project),
          includeWorkspace && (!isSplitProjectRecord(record) || record.workspace !== null)
        )
      : toRecoveryArchive(record, includeWorkspace);

  return exportProjectDocument(record.name, record.project_id, archived, record.minimum_canvas_schema_version, {
    ...options,
    owner,
  });
};

export const exportOpenProject = async (
  requested: Project,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const open = getOpenProject(requested.id);
  // The file is written from the live document, which holds the canvas pixels only after this barrier.
  await open?.flushPixels();
  assertAccountScopeCurrent(owner);
  const project = open?.current() ?? requested;
  const { serializeProjectDocumentV3 } = await import('./projectDocument');
  const document = serializeProjectDocumentV3(project);

  assertAccountScopeCurrent(owner);

  // Use the acknowledged server compatibility floor; offline fallback uses live canvas requirements.
  const record = open ? await readAcknowledgedProject(project.id, owner) : null;
  const minimumCanvasSchemaVersion = Math.max(
    getProjectCanvasSchemaRequirement(document),
    record?.minimum_canvas_schema_version ?? 1
  );

  return exportProjectDocument(
    project.name,
    project.id,
    toArchivedProject(document, options.includeWorkspace ?? true),
    minimumCanvasSchemaVersion,
    { ...options, owner }
  );
};

/** A file from before the mode existed would otherwise read as a stored project's Off. */
const withImportedViewerAutoSwitch = (settings: unknown): Record<string, unknown> => {
  const record = settings !== null && typeof settings === 'object' ? (settings as Record<string, unknown>) : {};

  return isViewerAutoSwitchMode(record.viewerAutoSwitch) ? record : { ...record, viewerAutoSwitch: 'progress' };
};

/** Every import stages an inbox board first, so the create claims it and carries the workspace in one request. */
export const importProjectFile = async (
  file: File,
  options: ProjectFileOptions = {}
): Promise<ProjectImportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const source = await readProjectDocument(file);
  const archivedDocument = source.format === 'invk' ? source.contents.projectDocument : source.projectDocument;
  let projectDocument = archivedDocument;
  if (archivedDocument.documentSchemaVersion === PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION) {
    const { joinProjectHalves } = await import('./projectHydration');
    const joined = joinProjectHalves(archivedDocument, source.format === 'invk' ? source.contents.workspace : null);

    if (joined.status === 'malformed') {
      throw new InvkFormatError('damaged', joined.reason);
    }
    projectDocument = joined.document;
  }

  assertAccountScopeCurrent(owner);

  const id = createProjectId();
  const name =
    typeof projectDocument.name === 'string' && projectDocument.name.trim()
      ? projectDocument.name.trim()
      : 'Imported project';
  // Strip selection on import because skipped references cannot be restored or reported.
  const stripped = stripInstallationState(projectDocument);
  const candidate = { ...stripped, id, name, settings: withImportedViewerAutoSwitch(stripped.settings) };
  // Validate through the reducer lazily to keep editor aggregates out of Launchpad imports.
  const { deserializeProjectDocument } = await import('./syncedPersistence');

  assertAccountScopeCurrent(owner);

  const loaded = deserializeProjectDocument(candidate);

  if (loaded.status === 'refused') {
    throw toInvkRefusalError(loaded.refused);
  }

  if (loaded.status !== 'loaded') {
    throw new InvkFormatError('damaged', 'The project document will not rehydrate.');
  }

  // Lazy: this file is shared with the Launchpad, which never loads the workflow core eagerly.
  const { stripProjectWorkflowSources } = await import('@workbench/projectWorkflows');
  const project = { ...loaded.project, workflows: stripProjectWorkflowSources(loaded.project.workflows) };

  const { serializeProjectDocument } = await import('./projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const archive = source.format === 'invk' ? source.contents : null;
  const [restoreMedia, memberBoards, { toTransferredProjectCreation }] = await Promise.all([
    import('./invk/restoreProjectMedia'),
    import('./invk/memberBoards'),
    import('./invk/transferCreation'),
  ]);
  const fontTransfer = archive?.fonts?.length && !options.skipEmbeddedFonts ? await import('./invk/fonts') : null;
  const fontTransport = fontTransfer ? (await import('./invk/fontTransport')).createFontArchiveTransport() : null;
  const fontLedger = fontTransfer?.createRestoredFontLedger() ?? null;

  if (fontTransfer && fontTransport && archive?.fonts) {
    await fontTransfer.preflightEmbeddedFonts(archive.fonts, fontTransport, owner.signal);
  }

  assertAccountScopeCurrent(owner);

  const archiveBoards = archive?.boardSnapshot?.boards ?? [];
  // Made before staging so a restore abandoned at any point can delete every staging board.
  const ledger = restoreMedia.createRestoredMediaLedger([]);
  let didCreateProject = false;
  let didAttemptProjectCreate = false;

  try {
    assertAccountScopeCurrent(owner);

    if (fontTransfer && fontTransport && fontLedger && archive?.fonts) {
      await fontTransfer.restoreEmbeddedFonts(
        archive.fonts,
        fontLedger,
        fontTransport,
        owner.signal,
        (completed, total) => options.onProgress?.({ completed, phase: 'restoring-fonts', total })
      );
      assertAccountScopeCurrent(owner);
    }

    const { boards: stagedBoards, inboxStagingBoardId } = await memberBoards.createStagingBoards(
      archiveBoards,
      name,
      ledger,
      owner.signal
    );

    assertAccountScopeCurrent(owner);

    const restored =
      archive === null
        ? null
        : await (async () => {
            const { restoreArchiveMedia } = await import('./invk/importProject');

            return restoreArchiveMedia(
              archive,
              { ledger, projectDocument: canonicalDocument, projectId: id, stagedBoards },
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

    const mediaDocument = restored === null ? canonicalDocument : remapAssetRefs(canonicalDocument, restored.mappings);
    const document =
      fontTransfer && fontLedger ? fontTransfer.remapFontReferences(mediaDocument, fontLedger.mappings) : mediaDocument;
    const minimumCanvasSchemaVersion = Math.max(
      getProjectCanvasSchemaRequirement(document),
      archive?.manifest.minimumCanvasSchemaVersion ?? 1
    );
    // One request, so the project never exists without the importer's workspace.
    didAttemptProjectCreate = true;
    const record = await createProjectSettled(
      {
        ...toTransferredProjectCreation(document as unknown as ProjectDocumentV3, inboxStagingBoardId),
        minimum_canvas_schema_version: minimumCanvasSchemaVersion,
        name,
        project_id: id,
      },
      owner
    );

    didCreateProject = true;
    assertAccountScopeCurrent(owner);
    // Never fatal from here on; lists fetched while staged would keep showing these boards in the Library.
    const boardIssues = await memberBoards
      .placeMemberBoards(stagedBoards, record.project_id, {
        onProgress: (completed, total) => options.onProgress?.({ completed, phase: 'placing-boards', total }),
        signal: owner.signal,
      })
      .finally(() => invalidateBoardLists(owner));
    assertAccountScopeCurrent(owner);
    upsertProjectSummary(
      {
        id: record.project_id,
        minimumCanvasSchemaVersion: record.minimum_canvas_schema_version,
        name: record.name,
        revision: record.revision,
      },
      owner
    );

    if (restored?.coverImageName) {
      recordProjectCover(record.project_id, restored.coverImageName, owner);
    }

    return {
      boardIssues,
      boardItemIssues: restored?.boardItemIssues ?? [],
      documentReferenceIssues: restored?.documentReferenceIssues ?? [],
      record,
    };
  } catch (error) {
    if (fontTransfer && fontTransport && fontLedger && isAccountScopeCurrent(owner)) {
      const rollback = () => fontTransfer.rollbackRestoredFonts(fontLedger, fontTransport, owner.signal);
      if (!didAttemptProjectCreate) {
        await rollback();
      } else {
        await restoreMedia.rollbackUnlessProjectExists(error, didCreateProject, owner, rollback);
      }
    }
    const rollback = () =>
      restoreMedia.rollbackRestoredMedia(ledger, { signal: owner.signal }).finally(() => invalidateBoardLists(owner));
    if (!didAttemptProjectCreate && isAccountScopeCurrent(owner)) {
      await rollback();
    } else {
      await restoreMedia.rollbackUnlessProjectExists(error, didCreateProject, owner, rollback);
    }

    throw error;
  }
};

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
