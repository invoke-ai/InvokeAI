import type { Project } from '@workbench/projectContracts';

import { downloadBlob } from '@platform/browser/downloadBlob';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';

import {
  createProject as apiCreateProject,
  getProject as apiGetProject,
  isProjectNotFoundError,
  type ProjectRecordDTO,
} from './api';
import { recordProjectCover } from './covers';
import { createProjectId } from './ids';
import { INVK_EXTENSION, InvkFormatError } from './invk/format';
import { upsertProjectSummary } from './library';
import { remapAssetRefs } from './projectAssets';

export const PROJECT_FILE_KIND = 'invokeai-project';
export const PROJECT_FILE_VERSION = 1;
export const LEGACY_PROJECT_FILE_EXTENSION = '.invokeproject.json';

export interface ProjectFile {
  document: Record<string, unknown>;
  exportedAt: string;
  kind: typeof PROJECT_FILE_KIND;
  version: typeof PROJECT_FILE_VERSION;
}

/** Kept for callers that still construct the JSON envelope shipped before `.invk`. */
export const buildProjectFile = (projectDocument: Record<string, unknown>): ProjectFile => ({
  document: projectDocument,
  exportedAt: new Date().toISOString(),
  kind: PROJECT_FILE_KIND,
  version: PROJECT_FILE_VERSION,
});

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
 * Export and import a project as an `.invk` archive.
 *
 * This module is the workflow; `./invk` is the format. The split matters
 * because the two have different reasons to change: the archive layout is a
 * compatibility surface shared with the previous frontend, while what an export
 * means here — which project, under whose account, landing where — is app
 * behavior.
 *
 * ### Why an archive and not a JSON file
 *
 * A project document references its pixels by server image name. Exporting the
 * document alone produced a file that opened perfectly on the machine that
 * wrote it and showed nothing but missing layers anywhere else. An `.invk`
 * carries the bytes, so the file is the project rather than a description of it.
 *
 * ### Import never overwrites
 *
 * An imported document gets a fresh id, never the one in the file. Two people
 * exchanging a project would otherwise collide the moment both saved, and
 * re-importing your own export would silently replace the original.
 *
 * The heavy halves — the ZIP codec and the workbench reducer — are both loaded
 * with `await import()`. The Launchpad offers Import and Export on a route that
 * never mounts the editor, and it should not pay for either until someone
 * actually picks a file.
 *
 * ### Both directions report as they go, and report what they lost
 *
 * A project file is the one operation here whose duration is set by how much
 * someone has drawn: a few hundred full-resolution layers is a few hundred
 * round trips and hundreds of megabytes. Silence for that long reads as a
 * broken button, so {@link ProjectFileProgress} is reported throughout.
 *
 * Both directions can also half-succeed. Export skips an asset the server will
 * not serve; import leaves a reference dangling when the archive did not carry
 * it and this server does not have it. Both were computed and discarded before,
 * which made a project that lost forty layers indistinguishable from a clean
 * round trip. They are returned now, and every call site says so.
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

export interface ProjectExportOutcome {
  /** Assets the server would not serve. Their references still ship. */
  missingAssetNames: string[];
  /** The name the archive was downloaded under. */
  fileName: string;
}

export interface ProjectImportOutcome {
  /** Referenced assets neither the archive nor this server has. */
  danglingAssetNames: string[];
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

/** A rejected create is safe to clean up only when its client-chosen id is confirmed absent. */
const isProjectConfirmedAbsent = async (projectId: string, owner: AccountScope): Promise<boolean> => {
  try {
    await apiGetProject(projectId, owner.signal);

    return false;
  } catch (error) {
    return isAccountScopeCurrent(owner) && isProjectNotFoundError(error);
  }
};

const exportProjectDocument = async (
  name: string,
  projectDocument: Record<string, unknown>,
  options: Required<Pick<ProjectFileOptions, 'owner'>> & ProjectFileOptions
): Promise<ProjectExportOutcome> => {
  const { executeInvkExport, planInvkExport } = await import('./invk/exportProject');
  const { onProgress, owner } = options;

  assertAccountScopeCurrent(owner);

  const plan = planInvkExport({
    appVersion: APP_VERSION,
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

  return { fileName: plan.fileName, missingAssetNames: result.missingAssetNames };
};

/** Export a closed project straight from its server record. */
export const exportLibraryProject = async (
  projectId: string,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const record = await apiGetProject(projectId, owner.signal);

  assertAccountScopeCurrent(owner);

  return exportProjectDocument(record.name, record.data, { ...options, owner });
};

/** Export an open project from its live in-memory document. */
export const exportOpenProject = async (
  project: Project,
  options: ProjectFileOptions = {}
): Promise<ProjectExportOutcome> => {
  const owner = options.owner ?? captureAccountScope();
  const { serializeProjectDocument } = await import('./projectDocument');

  assertAccountScopeCurrent(owner);

  return exportProjectDocument(project.name, serializeProjectDocument(project), { ...options, owner });
};

/**
 * Import an `.invk` as a new server project: restore its images, rewrite every
 * reference the server renamed, then create the project. Throws
 * {@link InvkFormatError} so callers can translate the reason rather than
 * surface an internal message.
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
  const candidate = { ...projectDocument, id, name };
  // Full validation rehydrates the document through the Workbench reducer, so
  // it is loaded here rather than imported: the Launchpad should not carry the
  // editor's aggregate state just to offer an Import button.
  const { deserializeProjectDocument } = await import('./syncedPersistence');

  assertAccountScopeCurrent(owner);

  const project = deserializeProjectDocument(candidate);

  if (!project) {
    throw new InvkFormatError('damaged', 'The project document will not rehydrate.');
  }

  const { serializeProjectDocument } = await import('./projectDocument');
  const canonicalDocument = serializeProjectDocument(project);
  const restored =
    source.format === 'invk'
      ? await (async () => {
          const { restoreArchiveAssets } = await import('./invk/importProject');

          return restoreArchiveAssets(
            { ...source.contents, projectDocument: canonicalDocument },
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
        })()
      : null;
  let didCreateProject = false;

  try {
    assertAccountScopeCurrent(owner);

    const document = restored === null ? canonicalDocument : remapAssetRefs(canonicalDocument, restored.mappings);
    const record = await apiCreateProject({ data: document, name, project_id: id }, owner.signal);

    didCreateProject = true;
    assertAccountScopeCurrent(owner);
    upsertProjectSummary({ id: record.project_id, name: record.name, revision: record.revision }, owner);

    if (restored?.coverImageName) {
      recordProjectCover(record.project_id, restored.coverImageName, owner);
    }

    return { danglingAssetNames: restored?.danglingAssetNames ?? [], record };
  } catch (error) {
    const canRollback =
      !didCreateProject &&
      restored !== null &&
      isAccountScopeCurrent(owner) &&
      (await isProjectConfirmedAbsent(id, owner));

    if (canRollback) {
      try {
        const { rollbackArchiveAssets } = await import('./invk/importProject');

        await rollbackArchiveAssets(restored.uploadedAssets, { signal: owner.signal });
      } catch {
        // Cleanup is best-effort. The create/scope failure is the actionable
        // result and must never be replaced by a failed delete request.
      }
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
