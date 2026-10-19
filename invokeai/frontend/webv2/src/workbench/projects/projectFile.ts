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

/** Imports receive fresh IDs; ZIP and reducer dependencies remain lazy. */

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
  /** Export only: whether the file carries the exporter's workspace. On unless turned off. */
  includeWorkspace?: boolean;
  skipEmbeddedFonts?: boolean;
}

/** What an archive's `project.json` and `workspace.json` carry, and the document schema the first is written in. */
interface ArchivedProject {
  content: Record<string, unknown>;
  documentSchemaVersion: number;
  workspace: Record<string, unknown> | null;
}

export interface ProjectExportOutcome extends ProjectTransferIssues {
  /** The name the archive was downloaded under. */
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

/** A whole document as an archive carries it: its content, and the exporter's workspace unless it stays out. */
const toArchivedProject = (document: ProjectDocumentV3, includeWorkspace: boolean): ArchivedProject => {
  const halves = splitProjectDocument(document);

  return {
    content: halves.content as unknown as Record<string, unknown>,
    documentSchemaVersion: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    workspace: includeWorkspace ? (halves.workspace as unknown as Record<string, unknown>) : null,
  };
};

/**
 * What the server holds of a project this build cannot read, verbatim, so it can still be recovered: content (schema 4
 * on, as a newer build may write it) beside the caller's workspace, or a whole document, whose workspace no build that
 * cannot read it can take out.
 */
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

/** Export a project from its server record, flushing it first — see {@link readAcknowledgedProject}. */
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

/** Export an open project from its live in-memory document. */
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

/**
 * Project creation commits remapped/staged media. Every import stages an inbox board first, an empty one for a file
 * without board entries, so the create claims it and carries the importer's workspace, open on it, in one request.
 */
export const importProjectFile = async (
  file: File,
  options: ProjectFileOptions = {}
): Promise<ProjectImportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const source = await readProjectDocument(file);
  const archivedDocument = source.format === 'invk' ? source.contents.projectDocument : source.projectDocument;
  let projectDocument = archivedDocument;
  // Project content (v3) is joined with the exporter's workspace, or without one with a new project's defaults.
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
  const candidate = { ...stripInstallationState(projectDocument), id, name };
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

  // An imported archive never inherits library write targets from the ids its workflows carry.
  // Lazy like the other document modules here: this file is shared with the Launchpad, which never loads the
  // workflow core eagerly.
  const { stripProjectWorkflowSources } = await import('@workbench/projectWorkflows');
  const project = { ...loaded.project, workflows: stripProjectWorkflowSources(loaded.project.workflows) };

  const { serializeProjectDocument } = await import('./projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const archive = source.format === 'invk' ? source.contents : null;
  // Every import stages at least its inbox board, which a failed create must undo, legacy JSON included.
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
  // Made before anything is staged: every staging board joins it as it is created, so a restore abandoned at any
  // point can delete them all.
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

    // One staging board per archive board; the inbox's rides the create, the rest are placed after.
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
    // The content and the importer's workspace (the archive's, or a new project's), open on the staged inbox, in one
    // request: the project never exists without it. A project created whole carries the workspace inside.
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
    // The project exists; its other boards can now belong to it. Reported, never fatal, from here on. Lists fetched
    // while they were still staged, the create's own refresh among them, would keep showing them in the Library.
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
    // The staging boards it deletes may already be listed: uploads to them announce themselves.
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
