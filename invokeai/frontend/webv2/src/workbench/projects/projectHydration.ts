import type { Project, ProjectLoadResult } from '@workbench/projectContracts';

import { loadWorkbenchProject, withAuthoritativeProjectBoard } from '@workbench/workbenchState';

import type { ProjectRecordDTO } from './api';

import {
  applyAuthoritativeProjectBoard,
  migrateProjectDocument,
  PROJECT_DOCUMENT_SCHEMA_VERSION,
} from './projectDocument';

const toFutureProjectDocumentRefusal = (
  data: Record<string, unknown>,
  projectId = typeof data.id === 'string' ? data.id : '',
  projectName = typeof data.name === 'string' ? data.name : ''
): ProjectLoadResult | null => {
  if (
    !Number.isSafeInteger(data.documentSchemaVersion) ||
    (data.documentSchemaVersion as number) <= PROJECT_DOCUMENT_SCHEMA_VERSION
  ) {
    return null;
  }

  return {
    refused: {
      projectId,
      projectName,
      raw: data,
      refusal: {
        raw: data,
        scope: 'project-document',
        status: 'unsupported-version',
        version: data.documentSchemaVersion as number,
      },
      source: 'project-document',
    },
    status: 'refused',
  };
};

export const deserializeProjectDocument = (data: Record<string, unknown>): ProjectLoadResult => {
  const futureDocument = toFutureProjectDocumentRefusal(data);
  if (futureDocument) {
    return futureDocument;
  }
  const migrated = migrateProjectDocument(data);
  if (migrated.status === 'malformed') {
    return {
      refused: {
        projectId: typeof data.id === 'string' ? data.id : '',
        projectName: typeof data.name === 'string' ? data.name : '',
        raw: data,
        refusal: { raw: data, reason: migrated.reason, scope: 'project-document', status: 'malformed' },
        source: 'project-document',
      },
      status: 'refused',
    };
  }

  const { documentSchemaVersion: _documentSchemaVersion, ...projectDocument } = migrated.document;
  const result = loadWorkbenchProject({
    ...projectDocument,
    undoRedo: { future: [], past: [] },
    workflowHistories: {},
  } as unknown as Project);

  return result.status === 'refused' ? { refused: { ...result.refused, raw: data }, status: 'refused' } : result;
};

export const deserializeProjectRecord = (record: ProjectRecordDTO): ProjectLoadResult => {
  const futureDocument = toFutureProjectDocumentRefusal(record.data, record.project_id, record.name);
  if (futureDocument) {
    return futureDocument;
  }
  const result = deserializeProjectDocument(
    applyAuthoritativeProjectBoard(record.data, record.board_id, { selectBoard: false })
  );

  if (result.status === 'loaded') {
    return { project: withAuthoritativeProjectBoard(result.project, record.board_id), status: 'loaded' };
  }
  if (result.status === 'refused') {
    return { refused: { ...result.refused, raw: record.data }, status: 'refused' };
  }
  return result;
};
