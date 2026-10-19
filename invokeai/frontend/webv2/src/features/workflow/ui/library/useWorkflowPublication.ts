import type { ProjectGraphState } from '@features/workflow/core/types';
import type { WorkflowPublicationController, WorkflowPublicationResult } from '@features/workflow/queries';

import { createWorkflowPublicationController } from '@features/workflow/queries';
import { useWorkflowNotifications, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { hasMultipleWorkflowReturnNodes, serializeWorkflowJson } from '@features/workflow/utility';
import { captureAccountScope, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { useCallback, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * One publication controller per account: every editor save and preview save goes through it, so a workflow is
 * never published twice at once and a lost answer is reconciled before it is sent again.
 */

let controller: WorkflowPublicationController | null = null;

export const getWorkflowPublicationController = (): WorkflowPublicationController => {
  controller ??= createWorkflowPublicationController();
  return controller;
};

/**
 * Publications that got no definitive answer, by project workflow. Each keeps its captured request and reserved id
 * so a resend after a lost response cannot create a second template. They outlive the dialog that reported them:
 * a new save for that workflow meets the unanswered one first, and returning to the workflow offers it once more.
 * An entry ends when the publication settles, when the user discards it, or with the account.
 */
const unresolvedPublications = new Map<string, { failure: WorkflowPublicationFailure; offered: boolean }>();

const unresolvedKey = (projectId: string, workflowId: string): string => `${projectId}\u0000${workflowId}`;

export const getUnresolvedWorkflowPublication = (
  projectId: string,
  workflowId: string
): WorkflowPublicationFailure | null =>
  unresolvedPublications.get(unresolvedKey(projectId, workflowId))?.failure ?? null;

/** The unanswered publication, the first time the user arrives at its workflow; later arrivals stay quiet. */
export const takeUnresolvedWorkflowPublicationOffer = (
  projectId: string,
  workflowId: string
): WorkflowPublicationFailure | null => {
  const entry = unresolvedPublications.get(unresolvedKey(projectId, workflowId));

  if (!entry || entry.offered) {
    return null;
  }

  entry.offered = true;
  return entry.failure;
};

/** Forgets an unanswered publication; a later save starts afresh, under a new reserved id. */
export const discardUnresolvedWorkflowPublication = (projectId: string, workflowId: string): void => {
  unresolvedPublications.delete(unresolvedKey(projectId, workflowId));
};

registerAccountOwnedResource({
  clear: () => {
    controller = null;
    unresolvedPublications.clear();
  },
  name: 'workflow-publication',
});

/** A settled publication: like the controller's result, except a failed one retries through the same settlement. */
export type WorkflowPublicationOutcome =
  | Exclude<WorkflowPublicationResult, { status: 'failed' }>
  | { status: 'failed'; message: string; retry: () => Promise<WorkflowPublicationOutcome> }
  | { status: 'rejected'; message: string };

export type WorkflowPublicationFailure = Extract<WorkflowPublicationOutcome, { status: 'failed' }>;

export interface PublishProjectWorkflowOptions {
  workflowId: string;
  /** Names the new template; the project workflow keeps its own name. */
  name: string;
}

export interface UpdateProjectWorkflowSourceOptions {
  workflowId: string;
  libraryWorkflowId: string;
  expectedRevision: number;
}

export interface WorkflowPublication {
  /** Creates a new template from the project workflow's current content and makes it the copy's update target. */
  saveAsNew: (options: PublishProjectWorkflowOptions) => Promise<WorkflowPublicationOutcome>;
  /** Replaces the template the project workflow came from, at exactly the revision the caller reviewed. */
  updateSource: (options: UpdateProjectWorkflowSourceOptions) => Promise<WorkflowPublicationOutcome>;
  /** Creates a template from a document that is not (yet) a project workflow, such as a compiled preview. */
  saveDocumentAsNew: (document: ProjectGraphState, name: string) => Promise<WorkflowPublicationOutcome>;
  /** Resends the captured request of a failed publication. */
  retry: (result: WorkflowPublicationFailure) => Promise<WorkflowPublicationOutcome>;
  isPublishing: (workflowId: string) => boolean;
}

/**
 * Where a settled publication lands. `adoptNameFrom` is the workflow's name when a save as new started: the workflow
 * takes the created template's name only while it still has that one.
 */
interface PublicationTarget {
  adoptNameFrom?: string;
  name: string;
  projectId: string;
  updateSource: boolean;
  workflowId: string;
}

export const useWorkflowPublication = (): WorkflowPublication => {
  const { t } = useTranslation();
  const { commands, project } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const publicationController = getWorkflowPublicationController();

  // Re-render when the in-flight set changes so `isPublishing` answers stay current.
  useSyncExternalStore(
    publicationController.subscribe,
    publicationController.getVersion,
    publicationController.getVersion
  );

  const validate = useCallback(
    (document: ProjectGraphState): string | null =>
      hasMultipleWorkflowReturnNodes(document) ? t('workflowLibrary.multipleWorkflowReturnNodes') : null,
    [t]
  );

  /** Applies a settled publication to the originating project workflow; a later project or account never sees it. */
  const settle = useCallback(
    (result: WorkflowPublicationResult, target: PublicationTarget): WorkflowPublicationOutcome => {
      if (result.status === 'published') {
        if (target.updateSource) {
          commands.setWorkflowSource(
            { projectId: target.projectId, workflowId: target.workflowId },
            { libraryWorkflowId: result.libraryWorkflowId, revision: result.revision }
          );
        }

        // A save as new turns the workflow into the template it created, so it takes that template's name; the
        // header and a later update confirmation then name the template the workflow is linked to. A name the user
        // chose meanwhile stands, and so does one that cannot be checked because another project is active now.
        if (target.adoptNameFrom !== undefined && result.kind === 'created') {
          const snapshot = project.getSnapshot();
          const current =
            snapshot.id === target.projectId
              ? snapshot.workflows.find((candidate) => candidate.document.id === target.workflowId)
              : undefined;

          if (current?.document.name === target.adoptNameFrom) {
            commands.renameWorkflow(target.workflowId, result.name, target.projectId);
          }
        }

        notify.success(
          t('workflowLibrary.saved'),
          result.kind === 'created'
            ? t('workflowLibrary.savedCreatedBody', { name: result.name })
            : t('workflowLibrary.savedUpdatedBody', { name: result.name })
        );
      } else if (result.status === 'invalid') {
        notify.error(t('workflowLibrary.saveFailed'), result.message);
      } else if (result.status === 'busy') {
        notify.info(t('workflowLibrary.saveBusy', { name: target.name }));
      }

      return result;
    },
    [commands, notify, project, t]
  );

  // A retry settles against the same target, so a save that lands on the second attempt still links the copy.
  const settleWithRetry = useCallback(
    function settleWithRetry(result: WorkflowPublicationResult, target: PublicationTarget): WorkflowPublicationOutcome {
      const key = unresolvedKey(target.projectId, target.workflowId);

      if (result.status === 'failed') {
        const failure: WorkflowPublicationFailure = {
          ...result,
          retry: async () => settleWithRetry(await result.retry(), target),
        };

        unresolvedPublications.set(key, { failure, offered: false });
        return failure;
      }

      if (result.status !== 'busy') {
        unresolvedPublications.delete(key);
      }

      return settle(result, target);
    },
    [settle]
  );

  const saveAsNew = useCallback(
    async ({ name, workflowId }: PublishProjectWorkflowOptions): Promise<WorkflowPublicationOutcome> => {
      const snapshot = project.getSnapshot();
      const entry = snapshot.workflows.find((candidate) => candidate.document.id === workflowId);

      if (!entry) {
        return { message: t('workflowLibrary.workflowGone'), status: 'rejected' };
      }

      const rejection = validate(entry.document);

      if (rejection) {
        notify.error(t('workflowLibrary.saveFailed'), rejection);
        return { message: rejection, status: 'rejected' };
      }

      const target = {
        adoptNameFrom: entry.document.name,
        name,
        projectId: snapshot.id,
        updateSource: true,
        workflowId,
      };
      const result = await publicationController.publish({
        destination: { kind: 'create', name },
        owner: captureAccountScope(),
        projectId: snapshot.id,
        workflow: serializeWorkflowJson(entry.document),
        workflowId,
      });

      return settleWithRetry(result, target);
    },
    [notify, project, publicationController, settleWithRetry, t, validate]
  );

  const updateSource = useCallback(
    async ({
      expectedRevision,
      libraryWorkflowId,
      workflowId,
    }: UpdateProjectWorkflowSourceOptions): Promise<WorkflowPublicationOutcome> => {
      const snapshot = project.getSnapshot();
      const entry = snapshot.workflows.find((candidate) => candidate.document.id === workflowId);

      if (!entry) {
        return { message: t('workflowLibrary.workflowGone'), status: 'rejected' };
      }

      const rejection = validate(entry.document);

      if (rejection) {
        notify.error(t('workflowLibrary.saveFailed'), rejection);
        return { message: rejection, status: 'rejected' };
      }

      const target = { name: entry.document.name, projectId: snapshot.id, updateSource: true, workflowId };
      const result = await publicationController.publish({
        destination: { expectedRevision, kind: 'update', libraryWorkflowId },
        owner: captureAccountScope(),
        projectId: snapshot.id,
        workflow: serializeWorkflowJson(entry.document),
        workflowId,
      });

      return settleWithRetry(result, target);
    },
    [notify, project, publicationController, settleWithRetry, t, validate]
  );

  const saveDocumentAsNew = useCallback(
    async (document: ProjectGraphState, name: string): Promise<WorkflowPublicationOutcome> => {
      const rejection = validate(document);

      if (rejection) {
        notify.error(t('workflowLibrary.saveFailed'), rejection);
        return { message: rejection, status: 'rejected' };
      }

      const projectId = project.getSnapshot().id;
      const target = { name, projectId, updateSource: false, workflowId: document.id };
      const result = await publicationController.publish({
        destination: { kind: 'create', name },
        owner: captureAccountScope(),
        projectId,
        workflow: serializeWorkflowJson(document),
        workflowId: document.id,
      });

      return settleWithRetry(result, target);
    },
    [notify, project, publicationController, settleWithRetry, t, validate]
  );

  // The failed result carries its captured request and, once settled here, its target.
  const retry = useCallback(
    (failed: WorkflowPublicationFailure): Promise<WorkflowPublicationOutcome> => failed.retry(),
    []
  );

  return {
    isPublishing: (workflowId) => publicationController.isPublishing(project.getSnapshot().id, workflowId),
    retry,
    saveAsNew,
    saveDocumentAsNew,
    updateSource,
  };
};
