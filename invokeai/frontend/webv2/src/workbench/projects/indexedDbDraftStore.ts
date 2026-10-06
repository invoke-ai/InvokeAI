import type { AccountScope } from '@platform/state/accountLifecycle';

import { createUuid } from '@platform/browser/randomUuid';

import {
  acquireAccountOwnedUnloadJournalDatabase,
  acquireAccountOwnedWorkbenchDatabase,
} from './accountOwnedWorkbenchDatabase';
import {
  clampProjectDraftLimit,
  combineProjectDraft,
  createUnavailableProjectDraftStore,
  decideUnloadJournalEntry,
  doProjectDraftPartsMatch,
  getProjectDraftSummary,
  getUtf8ByteSize,
  getCopySourceProjectName,
  isProjectDraft,
  isProjectDraftInput,
  isProjectDraftBody,
  isProjectDraftMetadata,
  isProjectDraftWriterClaim,
  isSameProjectDraftGeneration,
  PROJECT_DRAFT_MAX_BYTES,
  PROJECT_DRAFT_PAGE_LIMIT,
  PROJECT_DRAFT_PROJECT_LIMIT,
  toConflictProjectDraft,
  toDirtyProjectDraft,
  toProjectDraftBody,
  toProjectDraftMetadata,
  toSchemaRefusedProjectDraft,
  withUnloadJournalSettled,
  type ProjectDraft,
  type ProjectDraftAdoptionResult,
  type ProjectDraftClaimResult,
  type ProjectDraftCopyReservationResult,
  type ProjectDraftCorruptDeleteResult,
  type ProjectDraftDeleteResult,
  type ProjectDraftGetResult,
  type ProjectDraftKey,
  type ProjectDraftListResult,
  type ProjectDraftMetadata,
  type ProjectDraftPageResult,
  type ProjectDraftRetargetAcknowledgeResult,
  type ProjectDraftRetargetHandoff,
  type ProjectDraftRetargetListResult,
  type ProjectDraftSettlementResult,
  type ProjectDraftStageResult,
  type ProjectDraftStartWriterResult,
  type ProjectDraftStore,
  type ProjectDraftSummary,
  type ProjectUnloadJournalKey,
  type ProjectUnloadJournalOutcome,
  type ProjectUnloadJournalReconcileResult,
  type ProjectUnloadJournalReconciliation,
  type ProjectUnloadJournalSettleResult,
  type RetargetAcknowledgedCopyOptions,
} from './draftStore';
import {
  isWorkbenchDatabaseAvailable,
  UNLOAD_JOURNAL_STORE,
  WORKBENCH_DRAFT_BODY_STORE,
  WORKBENCH_DRAFT_STORE,
  WORKBENCH_DRAFT_WRITER_STORE,
  type UnloadJournalDatabase,
  type WorkbenchDatabase,
} from './workbenchDatabase';

const DRAFT_STORES = [WORKBENCH_DRAFT_STORE, WORKBENCH_DRAFT_BODY_STORE, WORKBENCH_DRAFT_WRITER_STORE] as const;
const isQuotaError = (error: unknown): boolean =>
  typeof error === 'object' && error !== null && 'name' in error && error.name === 'QuotaExceededError';

type DraftTransform = (draft: ProjectDraft) => ProjectDraft;
type ReadDraftResult =
  | ({ metadataRevision: number } & Extract<ProjectDraftGetResult, { kind: 'found' }>)
  | Exclude<ProjectDraftGetResult, { kind: 'found' }>;

const observeTransaction = <T extends { done: Promise<unknown> }>(transaction: T): T => {
  void transaction.done.catch(() => undefined);
  return transaction;
};

export const createIndexedDbProjectDraftStore = (
  database: WorkbenchDatabase,
  {
    journalDatabase = null,
    maxDraftBytes = PROJECT_DRAFT_MAX_BYTES,
  }: {
    /** Where unload journal entries live; without it, journaling and reconciliation report unavailable. */
    journalDatabase?: UnloadJournalDatabase | null;
    maxDraftBytes?: number;
  } = {}
): ProjectDraftStore => {
  let isClosed = false;
  let isUnavailable = false;

  const markUnavailable = (): void => {
    isUnavailable = true;
  };
  const canUseDatabase = (): boolean => !isClosed && !isUnavailable && isWorkbenchDatabaseAvailable(database);
  const mutate = async <T>(operation: () => Promise<T>, unavailable: T, quota = unavailable): Promise<T> => {
    if (!canUseDatabase()) {
      return unavailable;
    }
    try {
      return await operation();
    } catch (error) {
      if (isQuotaError(error)) {
        return quota;
      }
      markUnavailable();
      return unavailable;
    }
  };
  /** One writer's entries for a lineage through `generation`, or below it when `exclusive`. */
  const writerJournalRange = (
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    generation: number,
    exclusive: boolean
  ): IDBKeyRange =>
    IDBKeyRange.bound(
      [projectId, editorSessionId, writerToken, Number.NEGATIVE_INFINITY],
      [projectId, editorSessionId, writerToken, generation],
      false,
      exclusive
    );
  /**
   * Only a connection closed from outside is unavailable for good. Any other failure is the one operation's: the next
   * journal operation tries again, so one aborted deletion does not switch reload protection off for the session.
   */
  const canUseJournal = (): boolean =>
    journalDatabase !== null && !isClosed && isWorkbenchDatabaseAvailable(journalDatabase);
  /** Reads nothing first, so it never waits on a renderer round trip that an unloading page cannot make. */
  const deleteJournalKeys = async (query: IDBKeyRange | ProjectUnloadJournalKey): Promise<boolean> => {
    if (!canUseJournal()) {
      return false;
    }
    try {
      const transaction = observeTransaction(journalDatabase!.transaction(UNLOAD_JOURNAL_STORE, 'readwrite'));
      void transaction.store.delete(query).catch(() => undefined);
      transaction.commit();
      await transaction.done;
      return true;
    } catch {
      return false;
    }
  };
  const readDraft = (projectId: string, editorSessionId: string): Promise<ReadDraftResult> =>
    mutate<ReadDraftResult>(
      async () => {
        const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readonly'));
        const key: ProjectDraftKey = [projectId, editorSessionId];
        const [metadata, body, claim] = await Promise.all([
          transaction.objectStore(WORKBENCH_DRAFT_STORE).get(key),
          transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE).get(key),
          transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE).get(key),
        ]);
        await transaction.done;
        if (metadata === undefined) {
          if (body !== undefined || (claim !== undefined && !isProjectDraftWriterClaim(claim))) {
            return { kind: 'corrupt' };
          }
          return claim?.state === 'fenced' &&
            claim.retargetedToProjectId !== undefined &&
            claim.retargetedToRevision !== undefined
            ? {
                kind: 'retargeted',
                projectId: claim.retargetedToProjectId,
                revision: claim.retargetedToRevision,
                writerToken: claim.writerToken,
              }
            : claim
              ? { kind: 'empty', writerState: claim.state, writerToken: claim.writerToken }
              : { kind: 'missing' };
        }
        const draft = combineProjectDraft(metadata, body);
        return draft &&
          isProjectDraftWriterClaim(claim) &&
          claim.state === 'active' &&
          claim.writerToken === draft.writerToken &&
          claim.metadataRevision === metadata.metadataRevision
          ? { draft, kind: 'found', metadataRevision: metadata.metadataRevision }
          : { kind: 'corrupt' };
      },
      { kind: 'unavailable' }
    );
  const readRetargetedCopy = (
    options: RetargetAcknowledgedCopyOptions
  ): Promise<ProjectDraftSettlementResult | { kind: 'not-retargeted' }> =>
    mutate<ProjectDraftSettlementResult | { kind: 'not-retargeted' }>(
      async () => {
        const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readonly'));
        const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
        const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
        const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
        const sourceKey: ProjectDraftKey = [options.projectId, options.editorSessionId];
        const targetKey: ProjectDraftKey = [options.copyProjectId, options.editorSessionId];
        const [sourceMetadataKey, sourceBodyKey, sourceClaim, targetMetadata, targetBody, targetClaim] =
          await Promise.all([
            metadataStore.getKey(sourceKey),
            bodyStore.getKey(sourceKey),
            writerStore.get(sourceKey),
            metadataStore.get(targetKey),
            bodyStore.get(targetKey),
            writerStore.get(targetKey),
          ]);
        await transaction.done;
        if (sourceMetadataKey !== undefined || sourceBodyKey !== undefined) {
          return { kind: 'not-retargeted' };
        }
        if (!isProjectDraftWriterClaim(sourceClaim)) {
          return sourceClaim === undefined ? { kind: 'not-retargeted' } : { kind: 'corrupt' };
        }
        if (
          sourceClaim.state !== 'fenced' ||
          sourceClaim.retargetedToProjectId === undefined ||
          sourceClaim.retargetedToRevision === undefined
        ) {
          return { kind: 'not-retargeted' };
        }
        if (sourceClaim.writerToken !== options.writerToken) {
          return { kind: 'fenced' };
        }
        if (
          sourceClaim.retargetedToProjectId !== options.copyProjectId ||
          sourceClaim.retargetedToRevision !== options.acknowledgedRevision
        ) {
          return { kind: 'stale' };
        }
        if (
          !isProjectDraftWriterClaim(targetClaim) ||
          targetClaim.state !== 'active' ||
          targetClaim.writerToken !== options.writerToken ||
          targetClaim.metadataRevision < sourceClaim.metadataRevision
        ) {
          return targetClaim === undefined ? { kind: 'corrupt' } : { kind: 'fenced' };
        }
        if (targetMetadata === undefined) {
          return targetBody === undefined ? { draft: null, kind: 'retargeted' } : { kind: 'corrupt' };
        }
        const target = combineProjectDraft(targetMetadata, targetBody);
        return target &&
          target.writerToken === options.writerToken &&
          targetClaim.metadataRevision === targetMetadata.metadataRevision
          ? { draft: target, kind: 'retargeted' }
          : { kind: 'corrupt' };
      },
      { kind: 'unavailable' }
    );
  const settle = async (
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    sentGeneration: number,
    transform: DraftTransform,
    kind: 'marked' | 'rebased'
  ): Promise<ProjectDraftSettlementResult> => {
    const outcome = await mutate<ProjectDraftSettlementResult | { kind: 'read-marked' }>(
      async () => {
        const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
        const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
        const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
        const key: ProjectDraftKey = [projectId, editorSessionId];
        const [metadata, claim] = await Promise.all([
          metadataStore.get(key),
          transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE).get(key),
        ]);
        if (metadata === undefined) {
          await transaction.done;
          return { kind: 'missing' };
        }
        if (!isProjectDraftMetadata(metadata) || !isProjectDraftWriterClaim(claim)) {
          await transaction.done;
          return { kind: 'corrupt' };
        }
        if (claim.metadataRevision !== metadata.metadataRevision) {
          await transaction.done;
          return { kind: 'corrupt' };
        }
        const bodyKey = await bodyStore
          .index('byIntegrity')
          .getKey([projectId, editorSessionId, metadata.generation, metadata.documentByteSize]);
        if (bodyKey === undefined) {
          await transaction.done;
          return { kind: 'corrupt' };
        }
        if (claim.state !== 'active' || claim.writerToken !== writerToken || metadata.writerToken !== writerToken) {
          await transaction.done;
          return { kind: 'fenced' };
        }
        if (metadata.generation < sentGeneration) {
          await transaction.done;
          return { kind: 'stale' };
        }
        const { metadataRevision, ...draftMetadata } = metadata;
        const transformed = transform({ ...draftMetadata, documentJson: '' } as ProjectDraft);
        const nextMetadataRevision = metadataRevision + 1;
        await metadataStore.put(toProjectDraftMetadata(transformed, nextMetadataRevision));
        await transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE).put({
          ...claim,
          metadataRevision: nextMetadataRevision,
          updatedAt: Date.now(),
        });
        await transaction.done;
        return { kind: 'read-marked' };
      },
      { kind: 'unavailable' },
      { kind: 'quota' }
    );
    if (outcome.kind !== 'read-marked') {
      return outcome;
    }
    const read = await readDraft(projectId, editorSessionId);
    if (read.kind === 'found') {
      return read.draft.writerToken === writerToken ? { draft: read.draft, kind } : { kind: 'fenced' };
    }
    return read.kind === 'corrupt' || read.kind === 'unavailable' ? read : { kind: 'missing' };
  };

  return {
    get availability() {
      return canUseDatabase() ? 'available' : 'unavailable';
    },
    acknowledgeRetarget(projectId, editorSessionId, targetProjectId): Promise<ProjectDraftRetargetAcknowledgeResult> {
      return mutate<ProjectDraftRetargetAcknowledgeResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(WORKBENCH_DRAFT_WRITER_STORE, 'readwrite'));
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const claim = await transaction.store.get(key);
          if (
            !isProjectDraftWriterClaim(claim) ||
            claim.state !== 'fenced' ||
            claim.retargetedToProjectId !== targetProjectId ||
            claim.retargetedToRevision === undefined
          ) {
            await transaction.done;
            return { kind: 'stale' };
          }
          await transaction.store.delete(key);
          await transaction.done;
          return { kind: 'deleted' };
        },
        { kind: 'unavailable' }
      );
    },
    async adopt(projectId, fromEditorSessionId, toEditorSessionId, toWriterToken): Promise<ProjectDraftAdoptionResult> {
      const read = await readDraft(projectId, fromEditorSessionId);
      if (read.kind !== 'found') {
        return read.kind === 'corrupt' || read.kind === 'unavailable' ? read : { kind: 'missing' };
      }
      return mutate<ProjectDraftAdoptionResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const sourceKey: ProjectDraftKey = [projectId, fromEditorSessionId];
          const targetKey: ProjectDraftKey = [projectId, toEditorSessionId];
          const [sourceMetadata, sourceBodyKey, sourceClaim, targetMetadata, targetBodyKey, targetClaim] =
            await Promise.all([
              metadataStore.get(sourceKey),
              bodyStore.getKey(sourceKey),
              writerStore.get(sourceKey),
              metadataStore.get(targetKey),
              bodyStore.getKey(targetKey),
              writerStore.get(targetKey),
            ]);
          if (sourceMetadata === undefined) {
            await transaction.done;
            return { kind: 'missing' };
          }
          if (
            !isProjectDraftMetadata(sourceMetadata) ||
            sourceBodyKey === undefined ||
            !isProjectDraftWriterClaim(sourceClaim) ||
            sourceClaim.state !== 'active' ||
            sourceClaim.writerToken !== read.draft.writerToken ||
            sourceMetadata.writerToken !== read.draft.writerToken ||
            sourceClaim.metadataRevision !== sourceMetadata.metadataRevision
          ) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (sourceMetadata.metadataRevision !== read.metadataRevision) {
            await transaction.done;
            return { kind: 'occupied' };
          }
          if (targetClaim !== undefined && !isProjectDraftWriterClaim(targetClaim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (
            targetMetadata !== undefined ||
            targetBodyKey !== undefined ||
            (targetClaim?.state === 'active' && targetClaim.writerToken !== toWriterToken)
          ) {
            await transaction.done;
            return { kind: targetMetadata === undefined && targetBodyKey !== undefined ? 'corrupt' : 'occupied' };
          }
          const nextMetadataRevision = Math.max(sourceClaim.metadataRevision, targetClaim?.metadataRevision ?? 0) + 1;
          const target = { ...read.draft, editorSessionId: toEditorSessionId, writerToken: toWriterToken };
          await metadataStore.delete(sourceKey);
          await bodyStore.delete(sourceKey);
          await metadataStore.put(toProjectDraftMetadata(target, nextMetadataRevision));
          await bodyStore.put(toProjectDraftBody(target));
          await writerStore.put({
            editorSessionId: toEditorSessionId,
            metadataRevision: nextMetadataRevision,
            projectId,
            state: 'active',
            updatedAt: Date.now(),
            writerToken: toWriterToken,
          });
          await writerStore.put({
            adoptedByEditorSessionId: toEditorSessionId,
            editorSessionId: fromEditorSessionId,
            fenceReason: 'moved',
            metadataRevision: nextMetadataRevision,
            projectId,
            state: 'fenced',
            updatedAt: Date.now(),
            writerToken: read.draft.writerToken,
          });
          await transaction.done;
          return { kind: 'adopted' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
    },
    claimWriter(projectId, editorSessionId, expectedWriterToken, nextWriterToken): Promise<ProjectDraftClaimResult> {
      return mutate<ProjectDraftClaimResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const [metadata, claim] = await Promise.all([metadataStore.get(key), writerStore.get(key)]);
          if (claim === undefined) {
            await transaction.done;
            return { kind: 'missing' };
          }
          if (!isProjectDraftWriterClaim(claim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (claim.state === 'fenced' || claim.writerToken !== expectedWriterToken) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          if (metadata !== undefined) {
            if (
              !isProjectDraftMetadata(metadata) ||
              metadata.writerToken !== expectedWriterToken ||
              metadata.metadataRevision !== claim.metadataRevision
            ) {
              await transaction.done;
              return { kind: 'corrupt' };
            }
            const bodyKey = await bodyStore
              .index('byIntegrity')
              .getKey([projectId, editorSessionId, metadata.generation, metadata.documentByteSize]);
            if (bodyKey === undefined) {
              await transaction.done;
              return { kind: 'corrupt' };
            }
            const nextMetadataRevision = claim.metadataRevision + 1;
            await metadataStore.put({
              ...metadata,
              metadataRevision: nextMetadataRevision,
              writerToken: nextWriterToken,
            });
          } else if ((await bodyStore.getKey(key)) !== undefined) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          await writerStore.put({
            ...claim,
            metadataRevision: claim.metadataRevision + 1,
            updatedAt: Date.now(),
            writerToken: nextWriterToken,
          });
          await transaction.done;
          return { kind: 'claimed' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
    },
    close() {
      isClosed = true;
    },
    delete(projectId, editorSessionId, writerToken, settledThroughGeneration) {
      return mutate<ProjectDraftDeleteResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const [metadata, bodyKey, claim] = await Promise.all([
            metadataStore.get(key),
            bodyStore.getKey(key),
            writerStore.get(key),
          ]);
          if (isProjectDraftWriterClaim(claim) && (claim.state === 'fenced' || claim.writerToken !== writerToken)) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          if (
            (claim !== undefined && !isProjectDraftWriterClaim(claim)) ||
            (metadata === undefined) !== (bodyKey === undefined) ||
            (metadata !== undefined &&
              (!isProjectDraftMetadata(metadata) ||
                claim === undefined ||
                metadata.writerToken !== writerToken ||
                metadata.metadataRevision !== claim.metadataRevision))
          ) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          await metadataStore.delete(key);
          await bodyStore.delete(key);
          if (isProjectDraftWriterClaim(claim)) {
            const bumped = { ...claim, metadataRevision: claim.metadataRevision + 1, updatedAt: Date.now() };
            await writerStore.put(
              settledThroughGeneration === undefined
                ? bumped
                : withUnloadJournalSettled(bumped, settledThroughGeneration)
            );
          }
          await transaction.done;
          return { kind: 'deleted' };
        },
        { kind: 'unavailable' }
      );
    },
    deleteCorrupt(projectId, editorSessionId): Promise<ProjectDraftCorruptDeleteResult> {
      return mutate<ProjectDraftCorruptDeleteResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const [metadata, body, claim] = await Promise.all([
            metadataStore.get(key),
            bodyStore.get(key),
            writerStore.get(key),
          ]);
          const draft = combineProjectDraft(metadata, body);
          const isValidDraft =
            draft !== null &&
            isProjectDraftWriterClaim(claim) &&
            claim.state === 'active' &&
            claim.writerToken === draft.writerToken &&
            claim.metadataRevision === (metadata as ProjectDraftMetadata).metadataRevision;
          const isValidEmpty =
            metadata === undefined && body === undefined && (claim === undefined || isProjectDraftWriterClaim(claim));
          if (isValidDraft || isValidEmpty) {
            await transaction.done;
            return { kind: 'not-corrupt' };
          }
          await metadataStore.delete(key);
          await bodyStore.delete(key);
          const rawClaim = claim && typeof claim === 'object' ? claim : undefined;
          const rawMetadata = metadata && typeof metadata === 'object' ? metadata : undefined;
          const storedClaimRevision = rawClaim && 'metadataRevision' in rawClaim ? rawClaim.metadataRevision : 0;
          const storedMetadataRevision =
            rawMetadata && 'metadataRevision' in rawMetadata ? rawMetadata.metadataRevision : 0;
          const metadataRevision =
            Math.max(
              typeof storedClaimRevision === 'number' &&
                Number.isSafeInteger(storedClaimRevision) &&
                storedClaimRevision > 0 &&
                storedClaimRevision < Number.MAX_SAFE_INTEGER
                ? storedClaimRevision
                : 0,
              typeof storedMetadataRevision === 'number' &&
                Number.isSafeInteger(storedMetadataRevision) &&
                storedMetadataRevision > 0 &&
                storedMetadataRevision < Number.MAX_SAFE_INTEGER
                ? storedMetadataRevision
                : 0
            ) + 1;
          const storedWriterToken = rawClaim && 'writerToken' in rawClaim ? rawClaim.writerToken : undefined;
          await writerStore.put({
            editorSessionId,
            fenceReason: 'corrupt-cleanup',
            metadataRevision,
            projectId,
            state: 'fenced',
            updatedAt: Date.now(),
            writerToken:
              typeof storedWriterToken === 'string' && storedWriterToken.length > 0 ? storedWriterToken : createUuid(),
          });
          await transaction.done;
          return { kind: 'deleted' };
        },
        { kind: 'unavailable' }
      );
    },
    async discardUnloadJournal(projectId, editorSessionId, writerToken, throughGeneration = Number.MAX_VALUE) {
      const deleted = await deleteJournalKeys(
        writerJournalRange(projectId, editorSessionId, writerToken, throughGeneration, false)
      );
      return { kind: deleted ? 'deleted' : 'unavailable' };
    },
    async get(projectId, editorSessionId): Promise<ProjectDraftGetResult> {
      const result = await readDraft(projectId, editorSessionId);
      return result.kind === 'found' ? { draft: result.draft, kind: 'found' } : result;
    },
    journalBeforeUnload({ entries, retired }) {
      if (!canUseJournal()) {
        return { kind: 'unavailable' };
      }
      try {
        const transaction = journalDatabase!.transaction(UNLOAD_JOURNAL_STORE, 'readwrite');
        const written = transaction.done.then(
          () => true,
          () => false
        );
        for (const [projectId, editorSessionId, writerToken, generation] of retired) {
          void transaction.store
            .delete(writerJournalRange(projectId, editorSessionId, writerToken, generation, false))
            .catch(() => undefined);
        }
        for (const entry of entries) {
          // Each entry replaces its writer's earlier ones for the lineage, so they stay one per lineage and writer.
          void transaction.store
            .delete(
              writerJournalRange(entry.projectId, entry.editorSessionId, entry.writerToken, entry.generation, true)
            )
            .catch(() => undefined);
          void transaction.store.put(entry).catch(() => undefined);
        }
        // An unloading document aborts transactions still waiting on it; one committed now outlives it. Nothing is
        // read first, because a read's callback never runs during unload.
        transaction.commit();
        return { kind: 'started', written };
      } catch {
        return { kind: 'unavailable' };
      }
    },
    list({ after, limit: requestedLimit } = {}): Promise<ProjectDraftPageResult> {
      return mutate<ProjectDraftPageResult>(
        async () => {
          const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PAGE_LIMIT);
          const transaction = observeTransaction(database.transaction(WORKBENCH_DRAFT_STORE, 'readonly'));
          const range = after ? IDBKeyRange.lowerBound(after, true) : undefined;
          const items: ProjectDraftSummary[] = [];
          let cursor = await transaction.store.openCursor(range);
          let nextCursor: ProjectDraftKey | null = null;
          while (cursor) {
            if (items.length === limit) {
              nextCursor = [items.at(-1)!.projectId, items.at(-1)!.editorSessionId];
              break;
            }
            items.push(getProjectDraftSummary(cursor.value, cursor.primaryKey));
            cursor = await cursor.continue();
          }
          await transaction.done;
          return { items, kind: 'available', nextCursor };
        },
        { kind: 'unavailable' }
      );
    },
    listForProject(projectId, { after, limit: requestedLimit } = {}): Promise<ProjectDraftListResult> {
      return mutate<ProjectDraftListResult>(
        async () => {
          const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PROJECT_LIMIT);
          const transaction = observeTransaction(database.transaction(WORKBENCH_DRAFT_STORE, 'readonly'));
          const lower: IDBValidKey[] = after ? [projectId, after] : [projectId];
          const range = IDBKeyRange.bound(lower, [projectId, []], after !== undefined, false);
          const items: ProjectDraftSummary[] = [];
          let cursor = await transaction.store.openCursor(range);
          let nextCursor: string | null = null;
          while (cursor) {
            if (items.length === limit) {
              nextCursor = items.at(-1)?.editorSessionId ?? null;
              break;
            }
            items.push(getProjectDraftSummary(cursor.value, cursor.primaryKey));
            cursor = await cursor.continue();
          }
          await transaction.done;
          return { items, kind: 'available', nextCursor };
        },
        { kind: 'unavailable' }
      );
    },
    listRetargets({ after, limit: requestedLimit } = {}): Promise<ProjectDraftRetargetListResult> {
      return mutate<ProjectDraftRetargetListResult>(
        async () => {
          const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PAGE_LIMIT);
          const transaction = observeTransaction(database.transaction(WORKBENCH_DRAFT_WRITER_STORE, 'readonly'));
          const items: ProjectDraftRetargetHandoff[] = [];
          let cursor = await transaction.store
            .index('byRetarget')
            .openCursor(after ? IDBKeyRange.lowerBound(after, true) : undefined);
          while (cursor && items.length <= limit) {
            const claim = cursor.value;
            if (
              isProjectDraftWriterClaim(claim) &&
              claim.state === 'fenced' &&
              claim.retargetedToProjectId !== undefined &&
              claim.retargetedToRevision !== undefined
            ) {
              items.push({
                editorSessionId: claim.editorSessionId,
                projectId: claim.projectId,
                revision: claim.retargetedToRevision,
                targetProjectId: claim.retargetedToProjectId,
                updatedAt: claim.updatedAt,
              });
            }
            cursor = await cursor.continue();
          }
          await transaction.done;
          const hasMore = items.length > limit;
          if (hasMore) {
            items.pop();
          }
          const last = items.at(-1);
          return {
            items,
            kind: 'available',
            nextCursor: hasMore && last ? [last.projectId, last.editorSessionId, last.targetProjectId] : null,
          };
        },
        { kind: 'unavailable' }
      );
    },
    async peekUnloadJournalProjectIds(limit) {
      if (!canUseJournal()) {
        return { kind: 'unavailable' };
      }
      try {
        const keys = await journalDatabase!.getAllKeys(UNLOAD_JOURNAL_STORE, null, limit);
        return { kind: 'available', projectIds: [...new Set(keys.map(([projectId]) => projectId))] };
      } catch {
        return { kind: 'unavailable' };
      }
    },
    async reconcileUnloadJournal(
      accountId,
      now,
      { isEditorSessionLive } = {}
    ): Promise<ProjectUnloadJournalReconcileResult> {
      const outcomes: ProjectUnloadJournalReconciliation[] = [];
      const liveEditorSessions = new Map<string, boolean>();
      let after: ProjectUnloadJournalKey | null = null;
      for (;;) {
        if (!canUseJournal() || !canUseDatabase()) {
          return { kind: 'unavailable' };
        }
        let keys: ProjectUnloadJournalKey[];
        try {
          keys = await journalDatabase!.getAllKeys(
            UNLOAD_JOURNAL_STORE,
            after ? IDBKeyRange.lowerBound(after, true) : null,
            PROJECT_DRAFT_PAGE_LIMIT
          );
        } catch {
          return { kind: 'unavailable' };
        }
        for (const journalKey of keys) {
          if (isEditorSessionLive) {
            let isLive = liveEditorSessions.get(journalKey[1]);
            if (isLive === undefined) {
              isLive = await isEditorSessionLive(journalKey[1]);
              liveEditorSessions.set(journalKey[1], isLive);
            }
            if (isLive) {
              outcomes.push({ editorSessionId: journalKey[1], outcome: 'live', projectId: journalKey[0] });
              continue;
            }
          }
          let entry: unknown;
          try {
            entry = await journalDatabase!.get(UNLOAD_JOURNAL_STORE, journalKey);
          } catch {
            return { kind: 'unavailable' };
          }
          if (entry === undefined) {
            // Another tab reconciled it, or its writer discarded it, since the keys were read.
            continue;
          }
          let replacedDraft: ProjectUnloadJournalReconciliation['replacedDraft'];
          const outcome = await mutate<ProjectUnloadJournalOutcome | 'unavailable'>(
            async () => {
              const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
              const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
              const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
              const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
              const key: ProjectDraftKey = [journalKey[0], journalKey[1]];
              const [metadata, body, claim] = await Promise.all([
                metadataStore.get(key),
                bodyStore.get(key),
                writerStore.get(key),
              ]);
              const validClaim = claim === undefined || isProjectDraftWriterClaim(claim) ? claim : 'corrupt';
              const draft = metadata === undefined ? null : combineProjectDraft(metadata, body);
              const isCurrentConsistent =
                draft !== null &&
                validClaim !== undefined &&
                validClaim !== 'corrupt' &&
                validClaim.state === 'active' &&
                validClaim.writerToken === draft.writerToken &&
                validClaim.metadataRevision === (metadata as ProjectDraftMetadata).metadataRevision;
              const decision = decideUnloadJournalEntry({
                accountId,
                claim: validClaim,
                current:
                  metadata === undefined
                    ? body === undefined
                      ? null
                      : 'corrupt'
                    : isCurrentConsistent
                      ? draft
                      : 'corrupt',
                entry,
                maxDocumentBytes: maxDraftBytes,
                now,
              });
              if (decision.kind === 'apply') {
                const activeClaim = validClaim === 'corrupt' ? undefined : validClaim;
                const nextMetadataRevision = (activeClaim?.metadataRevision ?? 0) + 1;
                await writerStore.put(
                  activeClaim
                    ? { ...activeClaim, metadataRevision: nextMetadataRevision, updatedAt: decision.draft.updatedAt }
                    : {
                        editorSessionId: decision.draft.editorSessionId,
                        metadataRevision: nextMetadataRevision,
                        projectId: decision.draft.projectId,
                        state: 'active',
                        updatedAt: decision.draft.updatedAt,
                        writerToken: decision.draft.writerToken,
                      }
                );
                await metadataStore.put(toProjectDraftMetadata(decision.draft, nextMetadataRevision));
                await bodyStore.put(toProjectDraftBody(decision.draft));
                if (isCurrentConsistent) {
                  replacedDraft = { documentJson: draft.documentJson, generation: draft.generation };
                }
              }
              await transaction.done;
              return decision.kind === 'apply' ? 'applied' : decision.reason;
            },
            'unavailable',
            'quota'
          );
          if (outcome === 'unavailable') {
            return { kind: 'unavailable' };
          }
          // A retained or unwritten entry waits for a later load; the rest are done. Removing an entry after its
          // draft committed is safe to repeat: a second pass finds that draft and discards the entry as superseded.
          if (outcome !== 'corrupt' && outcome !== 'quota' && !(await deleteJournalKeys(journalKey))) {
            return { kind: 'unavailable' };
          }
          outcomes.push({
            editorSessionId: journalKey[1],
            outcome,
            projectId: journalKey[0],
            ...(replacedDraft ? { replacedDraft } : {}),
          });
        }
        if (keys.length < PROJECT_DRAFT_PAGE_LIMIT) {
          return { kind: 'available', outcomes };
        }
        after = keys.at(-1)!;
      }
    },
    reserveCopyIdentity(projectId, editorSessionId, writerToken, proposed, replaceCopyProjectId) {
      return mutate<ProjectDraftCopyReservationResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const [metadata, claim] = await Promise.all([
            metadataStore.get(key),
            transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE).get(key),
          ]);
          if (metadata === undefined) {
            await transaction.done;
            return { kind: 'missing' };
          }
          if (!isProjectDraftMetadata(metadata) || !isProjectDraftWriterClaim(claim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          const body = await bodyStore.get(key);
          if (!isProjectDraftBody(body) || !doProjectDraftPartsMatch(metadata, body)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (claim.state !== 'active' || claim.writerToken !== writerToken || metadata.writerToken !== writerToken) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          if (claim.metadataRevision !== metadata.metadataRevision) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (replaceCopyProjectId !== undefined && metadata.copyProjectId !== replaceCopyProjectId) {
            await transaction.done;
            return { kind: 'stale' };
          }
          const reservation =
            replaceCopyProjectId === undefined && metadata.copyProjectId
              ? {
                  copyDocumentByteSize: metadata.copyDocumentByteSize!,
                  copyDocumentJson: body.copyDocumentJson!,
                  copyProjectGeneration: metadata.copyProjectGeneration!,
                  copyProjectId: metadata.copyProjectId,
                  copyProjectMinimumCanvasSchemaVersion: metadata.copyProjectMinimumCanvasSchemaVersion!,
                  copyProjectName: metadata.copyProjectName!,
                  copySourceProjectName:
                    metadata.copySourceProjectName ?? getCopySourceProjectName(metadata.copyProjectName!),
                }
              : proposed;
          const { copyDocumentJson, ...reservationMetadata } = reservation;
          const nextMetadataRevision = metadata.metadataRevision + 1;
          await metadataStore.put({
            ...metadata,
            ...reservationMetadata,
            metadataRevision: nextMetadataRevision,
          });
          await bodyStore.put({ ...body, copyDocumentJson });
          await transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE).put({
            ...claim,
            metadataRevision: nextMetadataRevision,
            updatedAt: Date.now(),
          });
          await transaction.done;
          return { ...reservation, kind: 'reserved' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
    },
    resumeSchemaRefused(projectId, editorSessionId, writerToken, generation) {
      return settle(
        projectId,
        editorSessionId,
        writerToken,
        generation,
        (draft) => toDirtyProjectDraft(draft, {}),
        'marked'
      );
    },
    async retargetAcknowledgedCopy(options: RetargetAcknowledgedCopyOptions) {
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const read = await readDraft(options.projectId, options.editorSessionId);
        if (read.kind === 'retargeted') {
          const replay = await readRetargetedCopy(options);
          return replay.kind === 'not-retargeted' ? { kind: 'stale' } : replay;
        }
        if (read.kind !== 'found') {
          if (read.kind === 'empty') {
            return { kind: read.writerToken === options.writerToken ? 'missing' : 'fenced' };
          }
          return read;
        }
        if (read.draft.writerToken !== options.writerToken) {
          return { kind: 'fenced' };
        }
        if (read.draft.generation < options.sentGeneration || read.draft.copyProjectId !== options.copyProjectId) {
          return { kind: 'stale' };
        }
        let retargeted: { documentByteSize: number; documentJson: string } | null = null;
        if (read.draft.generation > options.sentGeneration) {
          let documentJson: string;
          try {
            documentJson = options.retargetDocument(read.draft.documentJson);
          } catch {
            return { kind: 'corrupt' };
          }
          const documentByteSize = getUtf8ByteSize(documentJson);
          if (documentByteSize > maxDraftBytes) {
            return { kind: 'too-large' };
          }
          retargeted = { documentByteSize, documentJson };
        }
        const outcome = await mutate<ProjectDraftSettlementResult | { kind: 'retry' }>(
          async () => {
            const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
            const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
            const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
            const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
            const sourceKey: ProjectDraftKey = [options.projectId, options.editorSessionId];
            const targetKey: ProjectDraftKey = [options.copyProjectId, options.editorSessionId];
            const [currentMetadata, sourceClaim, targetMetadata, targetBodyKey, targetClaim] = await Promise.all([
              metadataStore.get(sourceKey),
              writerStore.get(sourceKey),
              metadataStore.get(targetKey),
              bodyStore.getKey(targetKey),
              writerStore.get(targetKey),
            ]);
            if (!isProjectDraftMetadata(currentMetadata)) {
              await transaction.done;
              return { kind: currentMetadata === undefined ? 'retry' : 'corrupt' };
            }
            if (
              currentMetadata.metadataRevision !== read.metadataRevision ||
              !isProjectDraftWriterClaim(sourceClaim) ||
              sourceClaim.state !== 'active' ||
              sourceClaim.writerToken !== options.writerToken ||
              sourceClaim.metadataRevision !== currentMetadata.metadataRevision ||
              currentMetadata.writerToken !== options.writerToken ||
              currentMetadata.copyProjectId !== options.copyProjectId
            ) {
              await transaction.done;
              return currentMetadata.writerToken === options.writerToken &&
                sourceClaim?.writerToken === options.writerToken
                ? { kind: 'retry' }
                : { kind: 'fenced' };
            }
            if (targetClaim !== undefined && !isProjectDraftWriterClaim(targetClaim)) {
              await transaction.done;
              return { kind: 'corrupt' };
            }
            if (
              targetMetadata !== undefined ||
              targetBodyKey !== undefined ||
              (targetClaim?.state === 'active' && targetClaim.writerToken !== options.writerToken)
            ) {
              await transaction.done;
              return { kind: targetMetadata === undefined && targetBodyKey !== undefined ? 'corrupt' : 'occupied' };
            }
            const draft =
              retargeted === null
                ? null
                : toDirtyProjectDraft(read.draft, {
                    baseRevision: options.acknowledgedRevision,
                    copyDocumentByteSize: undefined,
                    copyDocumentJson: undefined,
                    copyProjectId: undefined,
                    copyProjectGeneration: undefined,
                    copyProjectMinimumCanvasSchemaVersion: undefined,
                    copyProjectName: undefined,
                    copySourceProjectName: undefined,
                    ...retargeted,
                    projectId: options.copyProjectId,
                  });
            const nextMetadataRevision = Math.max(sourceClaim.metadataRevision, targetClaim?.metadataRevision ?? 0) + 1;
            await metadataStore.delete(sourceKey);
            await bodyStore.delete(sourceKey);
            if (draft) {
              await metadataStore.put(toProjectDraftMetadata(draft, nextMetadataRevision));
              await bodyStore.put(toProjectDraftBody(draft));
            }
            await writerStore.put({
              editorSessionId: options.editorSessionId,
              metadataRevision: nextMetadataRevision,
              projectId: options.copyProjectId,
              state: 'active',
              updatedAt: Date.now(),
              writerToken: options.writerToken,
            });
            await writerStore.put({
              adoptedByEditorSessionId: options.editorSessionId,
              editorSessionId: options.editorSessionId,
              fenceReason: 'moved',
              metadataRevision: nextMetadataRevision,
              projectId: options.projectId,
              retargetedToProjectId: options.copyProjectId,
              retargetedToRevision: options.acknowledgedRevision,
              state: 'fenced',
              updatedAt: Date.now(),
              writerToken: read.draft.writerToken,
            });
            await transaction.done;
            return { draft, kind: 'retargeted' };
          },
          { kind: 'unavailable' },
          { kind: 'quota' }
        );
        if (outcome.kind !== 'retry') {
          return outcome;
        }
      }
      return { kind: 'stale' };
    },
    async settleAcknowledgement(
      projectId,
      editorSessionId,
      writerToken,
      sentGeneration,
      acknowledgedRevision,
      acknowledgedMinimumCanvasSchemaVersion
    ) {
      const outcome = await mutate<ProjectDraftSettlementResult | { kind: 'read-rebased' }>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
          const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const [metadata, claim] = await Promise.all([metadataStore.get(key), writerStore.get(key)]);
          if (metadata === undefined) {
            // Nothing to settle, but the writer's journal through this generation is obsolete all the same.
            if (isProjectDraftWriterClaim(claim) && claim.state === 'active' && claim.writerToken === writerToken) {
              await writerStore.put(withUnloadJournalSettled(claim, sentGeneration));
            }
            await transaction.done;
            return { kind: 'missing' };
          }
          if (!isProjectDraftMetadata(metadata) || !isProjectDraftWriterClaim(claim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          const bodyKey = await bodyStore
            .index('byIntegrity')
            .getKey([projectId, editorSessionId, metadata.generation, metadata.documentByteSize]);
          if (bodyKey === undefined) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (claim.state !== 'active' || claim.writerToken !== writerToken || metadata.writerToken !== writerToken) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          if (claim.metadataRevision !== metadata.metadataRevision) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (metadata.generation <= sentGeneration) {
            await writerStore.put(
              withUnloadJournalSettled(
                { ...claim, metadataRevision: claim.metadataRevision + 1, updatedAt: Date.now() },
                sentGeneration
              )
            );
            await metadataStore.delete(key);
            await bodyStore.delete(key);
            await transaction.done;
            return { kind: 'deleted' };
          }
          const nextMetadataRevision = metadata.metadataRevision + 1;
          await metadataStore.put({
            ...metadata,
            baseMinimumCanvasSchemaVersion:
              acknowledgedMinimumCanvasSchemaVersion ?? metadata.baseMinimumCanvasSchemaVersion,
            baseRevision: acknowledgedRevision,
            metadataRevision: nextMetadataRevision,
          });
          await writerStore.put(
            withUnloadJournalSettled(
              { ...claim, metadataRevision: nextMetadataRevision, updatedAt: Date.now() },
              sentGeneration
            )
          );
          await transaction.done;
          return { kind: 'read-rebased' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
      if (outcome.kind !== 'read-rebased') {
        return outcome;
      }
      const read = await readDraft(projectId, editorSessionId);
      if (read.kind === 'found') {
        return read.draft.writerToken === writerToken ? { draft: read.draft, kind: 'rebased' } : { kind: 'fenced' };
      }
      return read.kind === 'corrupt' || read.kind === 'unavailable' ? read : { kind: 'missing' };
    },
    settleConflict(projectId, editorSessionId, writerToken, sentGeneration, conflict) {
      return settle(
        projectId,
        editorSessionId,
        writerToken,
        sentGeneration,
        (draft) => toConflictProjectDraft(draft, conflict),
        'marked'
      );
    },
    settleSchemaRefusal(projectId, editorSessionId, writerToken, sentGeneration, refusal) {
      return settle(
        projectId,
        editorSessionId,
        writerToken,
        sentGeneration,
        (draft) => toSchemaRefusedProjectDraft(draft, refusal),
        'marked'
      );
    },
    async settleUnloadJournal(projectId, editorSessionId, writerToken, throughGeneration) {
      const settled = await mutate<ProjectUnloadJournalSettleResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const [metadataKey, claim] = await Promise.all([
            transaction.objectStore(WORKBENCH_DRAFT_STORE).getKey(key),
            writerStore.get(key),
          ]);
          if (claim === undefined ? metadataKey !== undefined : !isProjectDraftWriterClaim(claim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (claim && (claim.state !== 'active' || claim.writerToken !== writerToken)) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          await writerStore.put(
            withUnloadJournalSettled(
              claim ?? {
                editorSessionId,
                metadataRevision: 1,
                projectId,
                state: 'active',
                updatedAt: Date.now(),
                writerToken,
              },
              throughGeneration
            )
          );
          await transaction.done;
          return { kind: 'settled' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
      if (settled.kind === 'settled') {
        await deleteJournalKeys(writerJournalRange(projectId, editorSessionId, writerToken, throughGeneration, false));
      }
      return settled;
    },
    async stage(input): Promise<ProjectDraftStageResult> {
      if (!canUseDatabase()) {
        return { kind: 'unavailable' };
      }
      if (!isProjectDraftInput(input)) {
        return { kind: 'corrupt' };
      }
      const documentByteSize = getUtf8ByteSize(input.documentJson);
      if (documentByteSize > maxDraftBytes) {
        return { kind: 'too-large' };
      }
      try {
        const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
        const metadataStore = transaction.objectStore(WORKBENCH_DRAFT_STORE);
        const bodyStore = transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE);
        const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
        const key: ProjectDraftKey = [input.projectId, input.editorSessionId];
        const [currentMetadata, claim] = await Promise.all([metadataStore.get(key), writerStore.get(key)]);
        if (isProjectDraftWriterClaim(claim) && (claim.state === 'fenced' || claim.writerToken !== input.writerToken)) {
          await transaction.done;
          return { kind: 'fenced' };
        }
        if (
          (claim !== undefined && !isProjectDraftWriterClaim(claim)) ||
          (currentMetadata !== undefined && !isProjectDraftMetadata(currentMetadata)) ||
          (currentMetadata !== undefined &&
            (claim === undefined ||
              currentMetadata.writerToken !== input.writerToken ||
              currentMetadata.metadataRevision !== claim.metadataRevision))
        ) {
          await transaction.done;
          return { kind: 'corrupt' };
        }
        const currentBodyKey = currentMetadata
          ? await bodyStore
              .index('byIntegrity')
              .getKey([
                input.projectId,
                input.editorSessionId,
                currentMetadata.generation,
                currentMetadata.documentByteSize,
              ])
          : await bodyStore.getKey(key);
        if ((currentMetadata === undefined) !== (currentBodyKey === undefined)) {
          await transaction.done;
          return { kind: 'corrupt' };
        }
        let draft: ProjectDraft;
        if (currentMetadata) {
          if (currentMetadata.generation > input.generation) {
            await transaction.done;
            return { kind: 'stale' };
          }
          const needsCurrentBody =
            currentMetadata.generation === input.generation || currentMetadata.copyDocumentByteSize !== undefined;
          const currentBody = needsCurrentBody ? await bodyStore.get(key) : undefined;
          const currentDraft = needsCurrentBody ? combineProjectDraft(currentMetadata, currentBody) : null;
          if (needsCurrentBody && !currentDraft) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (currentMetadata.generation === input.generation) {
            await transaction.done;
            return {
              kind: isSameProjectDraftGeneration(currentDraft!, input) ? 'replayed' : 'generation-conflict',
            };
          }
          const { metadataRevision: _metadataRevision, ...metadata } = currentMetadata;
          draft = {
            ...metadata,
            ...(currentDraft?.copyDocumentJson === undefined
              ? {}
              : { copyDocumentJson: currentDraft.copyDocumentJson }),
            documentByteSize,
            documentJson: input.documentJson,
            documentSchemaVersion: input.documentSchemaVersion,
            generation: input.generation,
            updatedAt: input.updatedAt,
          } as ProjectDraft;
          if (!isProjectDraft(draft)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
        } else {
          draft = { ...input, documentByteSize, state: 'dirty' };
        }
        const nextMetadataRevision = (claim?.metadataRevision ?? 0) + 1;
        if (claim === undefined) {
          await writerStore.put({
            editorSessionId: input.editorSessionId,
            metadataRevision: nextMetadataRevision,
            projectId: input.projectId,
            state: 'active',
            updatedAt: input.updatedAt,
            writerToken: input.writerToken,
          });
        } else {
          await writerStore.put({ ...claim, metadataRevision: nextMetadataRevision, updatedAt: input.updatedAt });
        }
        await metadataStore.put(toProjectDraftMetadata(draft, nextMetadataRevision));
        await bodyStore.put(toProjectDraftBody(draft));
        await transaction.done;
        return { kind: 'stored' };
      } catch (error) {
        if (!isQuotaError(error)) {
          markUnavailable();
        }
        return { kind: isQuotaError(error) ? 'quota' : 'unavailable' };
      }
    },
    startFreshWriter(
      projectId,
      editorSessionId,
      expectedWriterToken,
      nextWriterToken
    ): Promise<ProjectDraftStartWriterResult> {
      return mutate<ProjectDraftStartWriterResult>(
        async () => {
          const transaction = observeTransaction(database.transaction(DRAFT_STORES, 'readwrite'));
          const key: ProjectDraftKey = [projectId, editorSessionId];
          const writerStore = transaction.objectStore(WORKBENCH_DRAFT_WRITER_STORE);
          const [metadataKey, bodyKey, claim] = await Promise.all([
            transaction.objectStore(WORKBENCH_DRAFT_STORE).getKey(key),
            transaction.objectStore(WORKBENCH_DRAFT_BODY_STORE).getKey(key),
            writerStore.get(key),
          ]);
          if ((metadataKey === undefined) !== (bodyKey === undefined)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (metadataKey !== undefined) {
            await transaction.done;
            return { kind: 'occupied' };
          }
          if (claim !== undefined && !isProjectDraftWriterClaim(claim)) {
            await transaction.done;
            return { kind: 'corrupt' };
          }
          if (claim ? claim.writerToken !== expectedWriterToken : expectedWriterToken !== null) {
            await transaction.done;
            return { kind: 'fenced' };
          }
          await writerStore.put({
            editorSessionId,
            metadataRevision: (claim?.metadataRevision ?? 0) + 1,
            projectId,
            state: 'active',
            updatedAt: Date.now(),
            writerToken: nextWriterToken,
          });
          await transaction.done;
          return { kind: 'started' };
        },
        { kind: 'unavailable' },
        { kind: 'quota' }
      );
    },
  };
};

export const createAccountOwnedProjectDraftStore = async (
  owner: AccountScope,
  dependencies: NonNullable<Parameters<typeof acquireAccountOwnedWorkbenchDatabase>[1]> & {
    openJournalDatabase?: (storageSuffix: string) => Promise<UnloadJournalDatabase>;
  } = {}
): Promise<ProjectDraftStore> => {
  // Without its own database, recovery works as before; only the unload journal is unavailable.
  const [leaseResult, journalLeaseResult] = await Promise.allSettled([
    acquireAccountOwnedWorkbenchDatabase(owner, dependencies),
    acquireAccountOwnedUnloadJournalDatabase(owner, {
      ...(dependencies.deleteDatabase ? { deleteDatabase: dependencies.deleteDatabase } : {}),
      ...(dependencies.openJournalDatabase ? { openDatabase: dependencies.openJournalDatabase } : {}),
    }),
  ]);
  const journalLease = journalLeaseResult.status === 'fulfilled' ? journalLeaseResult.value : null;
  if (leaseResult.status === 'rejected') {
    journalLease?.release();
    throw leaseResult.reason;
  }
  const lease = leaseResult.value;
  if (!lease) {
    journalLease?.release();
    return createUnavailableProjectDraftStore();
  }
  const ownedStore = createIndexedDbProjectDraftStore(lease.database, {
    journalDatabase: journalLease?.database ?? null,
  });
  const wrapStore = (): ProjectDraftStore => {
    let isReleased = false;
    return {
      ...ownedStore,
      get availability() {
        return ownedStore.availability;
      },
      close() {
        if (isReleased) {
          return;
        }
        isReleased = true;
        ownedStore.close();
        lease.release();
        journalLease?.release();
      },
    };
  };
  return wrapStore();
};
