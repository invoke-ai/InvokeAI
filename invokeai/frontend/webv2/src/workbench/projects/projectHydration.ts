import type { Project, ProjectLoadResult } from '@workbench/projectContracts';

import { createDraftProject, loadWorkbenchProject, withAuthoritativeProjectBoard } from '@workbench/workbenchState';

import type { ProjectWithWorkspaceDTO } from './api';

import { PROJECT_DOCUMENT_SCHEMA_VERSION, PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';
import {
  applyAuthoritativeProjectBoard,
  migrateProjectDocument,
  serializeProjectDocumentV3,
  type ProjectDocumentMigration,
} from './projectDocument';
import { findRepeatedWorkflowItem, joinProjectDocument, splitProjectDocument } from './projectHalves';

const toFutureProjectDocumentRefusal = (
  data: Record<string, unknown>,
  projectId = typeof data.id === 'string' ? data.id : '',
  projectName = typeof data.name === 'string' ? data.name : '',
  newestReadable = PROJECT_DOCUMENT_SCHEMA_VERSION
): ProjectLoadResult | null => {
  if (!Number.isSafeInteger(data.documentSchemaVersion) || (data.documentSchemaVersion as number) <= newestReadable) {
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

/** The workspace a project starts with, for a member who has not saved one in it. */
const createDefaultProjectWorkspace = () => {
  const { documentSchemaVersion: _schema, ...workspace } = splitProjectDocument(
    serializeProjectDocumentV3(createDraftProject([]))
  ).workspace;
  return workspace;
};

/** Whether a record holds project content (document schema 4 on) rather than a document saved whole. */
export const isSplitProjectRecord = (record: Pick<ProjectWithWorkspaceDTO, 'data'>): boolean =>
  record.data.documentSchemaVersion === PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION;

/** Content joined with a workspace, or with the defaults a new project starts with when there is none. */
export const joinProjectHalves = (
  content: Record<string, unknown>,
  workspace: Record<string, unknown> | null
): ProjectDocumentMigration => joinProjectDocument(content, workspace, createDefaultProjectWorkspace);

/**
 * The whole document a read holds: itself when saved whole, else its content joined with the caller's workspace, or
 * with the defaults a new project starts with before they have saved one. Split content that repeats an item id is
 * refused: what a lock or presence marker names must name one item (SF-12).
 */
export const joinProjectRecord = (record: ProjectWithWorkspaceDTO): ProjectDocumentMigration => {
  if (!isSplitProjectRecord(record)) {
    return { document: record.data, status: 'migrated' };
  }
  const joined = joinProjectHalves(record.data, record.workspace?.data ?? null);
  // A join always yields a workflow collection.
  const repeated =
    joined.status === 'migrated'
      ? findRepeatedWorkflowItem((joined.document.workflows as { entries: unknown[] }).entries)
      : null;
  return repeated === null ? joined : { reason: `Workflow ${repeated} repeats an item id.`, status: 'malformed' };
};

export const deserializeProjectRecord = (record: ProjectWithWorkspaceDTO): ProjectLoadResult => {
  const futureDocument = toFutureProjectDocumentRefusal(
    record.data,
    record.project_id,
    record.name,
    PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION
  );
  if (futureDocument) {
    return futureDocument;
  }
  const joined = joinProjectRecord(record);
  if (joined.status === 'malformed') {
    return {
      refused: {
        projectId: record.project_id,
        projectName: record.name,
        raw: record.data,
        refusal: { raw: record.data, reason: joined.reason, scope: 'project-document', status: 'malformed' },
        source: 'project-document',
      },
      status: 'refused',
    };
  }
  const whole = joined.document;
  const result = deserializeProjectDocument(
    applyAuthoritativeProjectBoard(whole, record.board_id, { selectBoard: false })
  );

  if (result.status === 'loaded') {
    return { project: withAuthoritativeProjectBoard(result.project, record.board_id), status: 'loaded' };
  }
  if (result.status === 'refused') {
    return { refused: { ...result.refused, raw: record.data }, status: 'refused' };
  }
  return result;
};
