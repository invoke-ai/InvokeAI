import type { AccountScope } from '@platform/state/accountLifecycle';

import { createUuid } from '@platform/browser/randomUuid';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';

import type { WorkflowLibraryWriteRefusedError, WorkflowRecordDTO } from './api';

import {
  createLibraryWorkflowRecord,
  getLibraryWorkflowRecord,
  toWorkflowLibraryWriteRefusal,
  updateLibraryWorkflow,
} from './api';
import { invalidateWorkflowLibraryCache } from './libraryCache';

/**
 * The one way a project workflow reaches the library. A publication captures everything it needs up front, refuses
 * to overlap another for the same workflow, keeps its captured request and reserved id across retryable failures,
 * and reconciles an ambiguous answer against the destination record before sending again.
 */

export type WorkflowPublicationDestination =
  | { kind: 'create'; name: string }
  | { kind: 'update'; libraryWorkflowId: string; expectedRevision: number };

export interface WorkflowPublicationRequest {
  owner: AccountScope;
  projectId: string;
  workflowId: string;
  /** The serialized document as it stood when the user confirmed; later edits are not part of this publication. */
  workflow: Record<string, unknown>;
  destination: WorkflowPublicationDestination;
}

export type WorkflowPublicationResult =
  | { status: 'published'; kind: 'created' | 'updated'; libraryWorkflowId: string; revision: number; name: string }
  /** The template moved past the expected revision, or its revision was never known. */
  | { status: 'conflict'; libraryWorkflowId: string; currentRevision: number | null }
  /** The destination cannot be written: it is gone, someone else's, or bundled. Saving as new still works. */
  | { status: 'unavailable'; libraryWorkflowId: string; reason: 'missing' | 'forbidden' | 'bundled' }
  /** The server rejected the content itself. */
  | { status: 'invalid'; message: string }
  /** Nothing definitive came back; `retry` resends the same captured request under the same reserved id. */
  | { status: 'failed'; message: string; retry: () => Promise<WorkflowPublicationResult> }
  /** The account changed underneath the publication; nothing more will be reported. */
  | { status: 'cancelled' }
  /** Another publication of the same workflow is still running. */
  | { status: 'busy' };

export interface WorkflowPublicationDeps {
  createRecord: typeof createLibraryWorkflowRecord;
  getRecord: typeof getLibraryWorkflowRecord;
  invalidateCache: typeof invalidateWorkflowLibraryCache;
  reserveId: () => string;
  updateRecord: typeof updateLibraryWorkflow;
}

const PRODUCTION_DEPS: WorkflowPublicationDeps = {
  createRecord: createLibraryWorkflowRecord,
  getRecord: getLibraryWorkflowRecord,
  invalidateCache: invalidateWorkflowLibraryCache,
  reserveId: createUuid,
  updateRecord: updateLibraryWorkflow,
};

/** The fields the server keeps; anything else a payload carries is ignored on both sides. */
const CONTENT_KEYS = [
  'author',
  'contact',
  'description',
  'edges',
  'exposedFields',
  'form',
  'meta',
  'name',
  'nodes',
  'notes',
  'tags',
  'version',
] as const;

const canonicalize = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }

  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize((value as Record<string, unknown>)[key])])
    );
  }

  return value;
};

/** Whether a stored record holds the submitted content, ignoring key order and the record's own id. */
export const isWorkflowContentEquivalent = (
  submitted: Record<string, unknown>,
  stored: Record<string, unknown>
): boolean =>
  CONTENT_KEYS.every(
    (key) => JSON.stringify(canonicalize(submitted[key] ?? null)) === JSON.stringify(canonicalize(stored[key] ?? null))
  );

const toPublished = (
  kind: 'created' | 'updated',
  record: WorkflowRecordDTO
): Extract<WorkflowPublicationResult, { status: 'published' }> => ({
  kind,
  libraryWorkflowId: record.workflow_id,
  name: record.name,
  revision: record.revision,
  status: 'published',
});

const toRefusalResult = (
  refusal: WorkflowLibraryWriteRefusedError,
  libraryWorkflowId: string
): WorkflowPublicationResult => {
  switch (refusal.reason) {
    case 'revision-conflict':
      return { currentRevision: refusal.currentRevision, libraryWorkflowId, status: 'conflict' };
    case 'missing':
    case 'forbidden':
    case 'bundled':
      return { libraryWorkflowId, reason: refusal.reason, status: 'unavailable' };
    case 'id-conflict':
    case 'invalid':
      return { message: refusal.message, status: 'invalid' };
  }
};

export interface WorkflowPublicationController {
  publish(request: WorkflowPublicationRequest): Promise<WorkflowPublicationResult>;
  /** True while a publication of that workflow is in flight. */
  isPublishing(projectId: string, workflowId: string): boolean;
  /** Advances whenever the in-flight set changes; a stable snapshot for `useSyncExternalStore`. */
  getVersion(): number;
  subscribe(listener: () => void): () => void;
}

export const createWorkflowPublicationController = (
  overrides: Partial<WorkflowPublicationDeps> = {}
): WorkflowPublicationController => {
  const deps = { ...PRODUCTION_DEPS, ...overrides };
  const inFlight = new Set<string>();
  const listeners = new Set<() => void>();
  let version = 0;
  const keyOf = (projectId: string, workflowId: string): string => `${projectId}\u0000${workflowId}`;
  const notify = (): void => {
    version += 1;
    listeners.forEach((listener) => listener());
  };

  const settle = async (
    request: WorkflowPublicationRequest,
    send: () => Promise<WorkflowPublicationResult>
  ): Promise<WorkflowPublicationResult> => {
    const key = keyOf(request.projectId, request.workflowId);

    if (inFlight.has(key)) {
      return { status: 'busy' };
    }

    inFlight.add(key);
    notify();

    try {
      if (!isAccountScopeCurrent(request.owner)) {
        return { status: 'cancelled' };
      }

      const result = await send();

      if (!isAccountScopeCurrent(request.owner)) {
        return { status: 'cancelled' };
      }

      if (result.status === 'published') {
        deps.invalidateCache(result.libraryWorkflowId);
      }

      return result;
    } finally {
      inFlight.delete(key);
      notify();
    }
  };

  /** A failure that is not a server refusal: the request may or may not have landed. */
  const toFailure = (
    error: unknown,
    request: WorkflowPublicationRequest,
    retry: () => Promise<WorkflowPublicationResult>
  ): WorkflowPublicationResult => {
    if (!isAccountScopeCurrent(request.owner)) {
      return { status: 'cancelled' };
    }

    return { message: getApiErrorMessage(error, 'The library did not answer.'), retry, status: 'failed' };
  };

  const create = (request: WorkflowPublicationRequest, name: string, reservedId: string) => {
    const attempt = (): Promise<WorkflowPublicationResult> =>
      settle(request, async () => {
        try {
          // The server matches a resend against the record the first send made, so a lost response never
          // creates a second template.
          const record = await deps.createRecord(
            { ...request.workflow, name },
            { reservedId, signal: request.owner.signal }
          );

          return toPublished('created', record);
        } catch (error) {
          const refusal = toWorkflowLibraryWriteRefusal(error);

          return refusal ? toRefusalResult(refusal, reservedId) : toFailure(error, request, attempt);
        }
      });

    return attempt();
  };

  const update = (request: WorkflowPublicationRequest, libraryWorkflowId: string, expectedRevision: number) => {
    // The template keeps its own name: an update replaces its content, and the project workflow's name stays
    // with the project. The content is fixed by the first read so a resend carries exactly what was sent before.
    let content: Record<string, unknown> | null = null;

    const send = async (record: WorkflowRecordDTO): Promise<WorkflowPublicationResult> => {
      content ??= { ...request.workflow, name: record.name };

      try {
        const updated = await deps.updateRecord(libraryWorkflowId, content, {
          expectedRevision,
          signal: request.owner.signal,
        });

        return toPublished('updated', updated);
      } catch (error) {
        const refusal = toWorkflowLibraryWriteRefusal(error);

        return refusal ? toRefusalResult(refusal, libraryWorkflowId) : toFailure(error, request, attempt);
      }
    };

    /**
     * Every attempt starts from the live record. The first read supplies the template's name; a read before a
     * resend tells whether the lost answer applied the write (the revision advanced by exactly one to this
     * content), left it untouched, or whether someone else moved the template.
     */
    const attempt = (): Promise<WorkflowPublicationResult> =>
      settle(request, async () => {
        let record: WorkflowRecordDTO;

        try {
          record = await deps.getRecord(libraryWorkflowId, request.owner.signal);
        } catch (error) {
          const refusal = toWorkflowLibraryWriteRefusal(error);

          return refusal ? toRefusalResult(refusal, libraryWorkflowId) : toFailure(error, request, attempt);
        }

        if (record.revision === expectedRevision) {
          return send(record);
        }

        if (
          content !== null &&
          record.revision === expectedRevision + 1 &&
          isWorkflowContentEquivalent(content, record.workflow)
        ) {
          return toPublished('updated', record);
        }

        return { currentRevision: record.revision, libraryWorkflowId, status: 'conflict' };
      });

    return attempt();
  };

  return {
    getVersion: () => version,
    isPublishing: (projectId, workflowId) => inFlight.has(keyOf(projectId, workflowId)),
    publish: (request) =>
      request.destination.kind === 'create'
        ? create(request, request.destination.name, deps.reserveId())
        : update(request, request.destination.libraryWorkflowId, request.destination.expectedRevision),
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};
