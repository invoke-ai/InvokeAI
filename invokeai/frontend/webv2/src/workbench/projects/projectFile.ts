import { createProjectId } from './ids';
import { upsertProjectSummary } from './library';
import { deserializeProjectDocument, serializeProjectDocument } from './syncedPersistence';
import { createProject as apiCreateProject, getProject as apiGetProject, type ProjectRecordDTO } from './api';
import type { Project } from '@workbench/types';

/**
 * The portable project file: a versioned envelope around the same document
 * shape the server stores. Export and import both reuse the sync layer's
 * serialize/deserialize pair, so the file format inherits its healing of
 * partial documents and there is exactly one definition of "a project
 * document" in the app.
 */

export const PROJECT_FILE_KIND = 'invokeai-project';
export const PROJECT_FILE_VERSION = 1;

export interface ProjectFile {
  kind: typeof PROJECT_FILE_KIND;
  version: typeof PROJECT_FILE_VERSION;
  exportedAt: string;
  document: Record<string, unknown>;
}

export const buildProjectFile = (projectDocument: Record<string, unknown>): ProjectFile => ({
  document: projectDocument,
  exportedAt: new Date().toISOString(),
  kind: PROJECT_FILE_KIND,
  version: PROJECT_FILE_VERSION,
});

/** Returns the embedded document, or null when the text is not a project file. */
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

const sanitizeFileName = (name: string): string => name.replace(/[^\w\- ]+/gu, '').trim() || 'project';

const downloadProjectFile = (name: string, projectDocument: Record<string, unknown>): void => {
  const blob = new Blob([JSON.stringify(buildProjectFile(projectDocument), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.download = `${sanitizeFileName(name)}.invokeproject.json`;
  anchor.click();
  URL.revokeObjectURL(url);
};

/** Export a closed project straight from its server record. */
export const exportLibraryProject = async (projectId: string): Promise<void> => {
  const record = await apiGetProject(projectId);

  downloadProjectFile(record.name, record.data);
};

/** Export an open project from its live in-memory document. */
export const exportOpenProject = (project: Project): void => {
  downloadProjectFile(project.name, serializeProjectDocument(project));
};

/**
 * Import a project file as a new server project. The document gets a fresh
 * id — never the one in the file — so an import can never collide with (and
 * overwrite) an existing project. Throws with a user-readable message.
 */
export const importProjectFile = async (file: File): Promise<ProjectRecordDTO> => {
  const projectDocument = parseProjectFile(await file.text());

  if (!projectDocument) {
    throw new Error('This file is not an Invoke project export.');
  }

  const id = createProjectId();
  const name =
    typeof projectDocument.name === 'string' && projectDocument.name.trim()
      ? projectDocument.name.trim()
      : 'Imported project';
  const document = { ...projectDocument, id, name };

  if (!deserializeProjectDocument(document)) {
    throw new Error('The project file is damaged and cannot be opened.');
  }

  const record = await apiCreateProject({ data: document, name, project_id: id });

  upsertProjectSummary({ id: record.project_id, name: record.name, revision: record.revision });

  return record;
};

/** Open the browser's file picker for a project file; null when dismissed. */
export const pickProjectFile = (): Promise<File | null> =>
  new Promise((resolve) => {
    const input = document.createElement('input');

    input.type = 'file';
    input.accept = 'application/json,.json';
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.oncancel = () => resolve(null);
    input.click();
  });
