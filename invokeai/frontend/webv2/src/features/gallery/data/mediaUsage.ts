/**
 * The server's media usage check (`POST /api/v1/gallery/usage`), loaded only by the deletion dialogs. A request names
 * at most the server's per-kind cap, so a larger selection is checked in batches whose answers merge into one.
 */

import type { components } from '@api/schema';
import type {
  GalleryMediaUsage,
  GalleryMediaUsageDocument,
  GalleryMediaUsageDocumentKind,
  GalleryMediaUsageNames,
  GalleryMediaUsageRequest,
} from '@features/gallery/core/mediaUsage';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';

import { toNameBatches } from './backend';

type GalleryMediaUsageWire = components['schemas']['GalleryMediaUsage'];
type GalleryMediaUsageNamesWire = Pick<
  components['schemas']['GalleryMediaUsageQueue'],
  'image_count' | 'image_names' | 'video_count' | 'video_names'
>;
type GalleryMediaUsageRequestWire = components['schemas']['GalleryMediaUsageRequest'];

const GALLERY_MEDIA_USAGE_DOCUMENT_KINDS: Readonly<Record<string, GalleryMediaUsageDocumentKind>> = {
  legacy_editor: 'legacyEditor',
  project: 'project',
  quarantined_project: 'quarantinedProject',
  workflow: 'workflow',
};

/** A malformed answer must fail the check, never read as "nothing uses it". */
const readUsageCount = (value: unknown, field: string): number => {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`Gallery media usage field "${field}" must be a non-negative integer.`);
  }

  return value;
};

const readUsageNames = (value: unknown, field: string): string[] => {
  if (!Array.isArray(value) || !value.every((name): name is string => typeof name === 'string')) {
    throw new TypeError(`Gallery media usage field "${field}" must be an array of strings.`);
  }

  return value;
};

/** A revision is only a hint for narrowing: one that is not a non-negative integer reads as unknown. */
const readUsageRevision = (value: unknown): number | null =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;

const mapGalleryMediaUsageNames = (wire: GalleryMediaUsageNamesWire, field: string): GalleryMediaUsageNames => ({
  imageCount: readUsageCount(wire.image_count, `${field}.image_count`),
  imageNames: readUsageNames(wire.image_names, `${field}.image_names`),
  videoCount: readUsageCount(wire.video_count, `${field}.video_count`),
  videoNames: readUsageNames(wire.video_names, `${field}.video_names`),
});

const mapGalleryMediaUsage = (body: GalleryMediaUsageWire): GalleryMediaUsage => {
  if (!Array.isArray(body.documents) || typeof body.queue !== 'object' || typeof body.other_accounts !== 'object') {
    throw new TypeError('Gallery media usage response is malformed.');
  }

  return {
    checkedImageCount: readUsageCount(body.checked_image_count, 'checked_image_count'),
    checkedVideoCount: readUsageCount(body.checked_video_count, 'checked_video_count'),
    documents: body.documents.map((document, index) => ({
      ...mapGalleryMediaUsageNames(document, `documents[${index}]`),
      id: String(document.id),
      // A kind added after this build still uses the media: it is listed, not dropped.
      kind: GALLERY_MEDIA_USAGE_DOCUMENT_KINDS[document.kind] ?? 'other',
      name: typeof document.name === 'string' && document.name.trim() !== '' ? document.name : null,
      revision: readUsageRevision(document.revision),
      workspaceRevision: readUsageRevision(document.workspace_revision),
    })),
    documentsTotal: readUsageCount(body.documents_total, 'documents_total'),
    inUseImageCount: readUsageCount(body.in_use_image_count, 'in_use_image_count'),
    inUseVideoCount: readUsageCount(body.in_use_video_count, 'in_use_video_count'),
    otherAccounts: {
      documentCount: readUsageCount(body.other_accounts.document_count, 'other_accounts.document_count'),
      imageCount: readUsageCount(body.other_accounts.image_count, 'other_accounts.image_count'),
      queueItemCount: readUsageCount(body.other_accounts.queue_item_count, 'other_accounts.queue_item_count'),
      videoCount: readUsageCount(body.other_accounts.video_count, 'other_accounts.video_count'),
    },
    queue: {
      ...mapGalleryMediaUsageNames(body.queue, 'queue'),
      itemCount: readUsageCount(body.queue.item_count, 'queue.item_count'),
    },
  };
};

const byDocumentOrder = (left: GalleryMediaUsageDocument, right: GalleryMediaUsageDocument): number =>
  left.kind.localeCompare(right.kind) ||
  (left.name ?? '').toLowerCase().localeCompare((right.name ?? '').toLowerCase()) ||
  left.id.localeCompare(right.id);

/** One batch's answer, and each listed document's identity as the server sent it (`kind:id`, kind unmapped). */
interface BatchAnswer {
  documentKeys: readonly string[];
  usage: GalleryMediaUsage;
}

/**
 * One answer for a check sent in batches. The batches check disjoint names, so name lists join and per-name counts
 * add. A document merges by the server's kind and id, keeping its revisions only when every batch that listed it read
 * the same ones (a save between batches leaves them unknown, so the host reads the server's copy). A batch past the
 * server's document listing cap may have left out a document another batch lists: such a document is marked
 * `namesIncomplete`, its revisions unknown. Counts that cannot be deduplicated across batches err high: documents a
 * batch left unlisted each count as another, and a job or another account's document naming media of several
 * batches counts once per batch.
 */
const mergeGalleryMediaUsage = (answers: readonly BatchAnswer[]): GalleryMediaUsage => {
  const documents = new Map<string, { batches: Set<number>; document: GalleryMediaUsageDocument }>();
  const overflowedBatches: number[] = [];
  let unlistedDocuments = 0;

  answers.forEach(({ documentKeys, usage }, batch) => {
    const unlisted = usage.documentsTotal - usage.documents.length;

    if (unlisted > 0) {
      overflowedBatches.push(batch);
      unlistedDocuments += unlisted;
    }
    usage.documents.forEach((document, index) => {
      const key = documentKeys[index]!;
      const entry = documents.get(key);

      if (!entry) {
        documents.set(key, {
          batches: new Set([batch]),
          document: { ...document, imageNames: [...document.imageNames], videoNames: [...document.videoNames] },
        });
        return;
      }

      const merged = entry.document;

      entry.batches.add(batch);
      merged.imageCount += document.imageCount;
      merged.imageNames.push(...document.imageNames);
      merged.videoCount += document.videoCount;
      merged.videoNames.push(...document.videoNames);
      merged.name ??= document.name;
      if (merged.revision !== document.revision || merged.workspaceRevision !== document.workspaceRevision) {
        merged.revision = null;
        merged.workspaceRevision = null;
      }
    });
  });

  const listed = [...documents.values()]
    .map(({ batches, document }) =>
      overflowedBatches.some((batch) => !batches.has(batch))
        ? { ...document, namesIncomplete: true as const, revision: null, workspaceRevision: null }
        : document
    )
    .sort(byDocumentOrder);
  const sum = (read: (usage: GalleryMediaUsage) => number) =>
    answers.reduce((total, { usage }) => total + read(usage), 0);

  return {
    checkedImageCount: sum((usage) => usage.checkedImageCount),
    checkedVideoCount: sum((usage) => usage.checkedVideoCount),
    documents: listed,
    documentsTotal: listed.length + unlistedDocuments,
    inUseImageCount: sum((usage) => usage.inUseImageCount),
    inUseVideoCount: sum((usage) => usage.inUseVideoCount),
    otherAccounts: {
      documentCount: sum((usage) => usage.otherAccounts.documentCount),
      imageCount: sum((usage) => usage.otherAccounts.imageCount),
      queueItemCount: sum((usage) => usage.otherAccounts.queueItemCount),
      videoCount: sum((usage) => usage.otherAccounts.videoCount),
    },
    queue: {
      imageCount: sum((usage) => usage.queue.imageCount),
      imageNames: answers.flatMap(({ usage }) => usage.queue.imageNames),
      itemCount: sum((usage) => usage.queue.itemCount),
      videoCount: sum((usage) => usage.queue.videoCount),
      videoNames: answers.flatMap(({ usage }) => usage.queue.videoNames),
    },
  };
};

/**
 * What still uses these items, or a board's items, per the server: the caller's projects, workflows and legacy editor
 * state, their waiting or running queue work, and other accounts' use without identities. Names the caller may not
 * delete are left out, as the delete endpoints skip them. Items past one request's cap are checked in batches, one
 * after another; any batch failing fails the check.
 */
export const checkGalleryMediaUsage = async (
  request: GalleryMediaUsageRequest,
  signal?: AbortSignal
): Promise<GalleryMediaUsage> => {
  const owner = captureAccountScope();
  const requestSignal = signal ? AbortSignal.any([signal, owner.signal]) : owner.signal;
  const ask = async (payload: GalleryMediaUsageRequestWire): Promise<BatchAnswer> => {
    requestSignal.throwIfAborted();
    const body = await apiFetchJson<GalleryMediaUsageWire>('/api/v1/gallery/usage', {
      body: JSON.stringify(payload),
      method: 'POST',
      signal: requestSignal,
    });

    assertAccountScopeCurrent(owner);
    const usage = mapGalleryMediaUsage(body);

    return { documentKeys: body.documents.map((document) => `${document.kind}:${document.id}`), usage };
  };

  if (request.kind === 'board') {
    return (await ask({ board_id: request.boardId })).usage;
  }

  const imageBatches = toNameBatches(request.refs.filter((ref) => ref.kind === 'image').map((ref) => ref.name));
  const videoBatches = toNameBatches(request.refs.filter((ref) => ref.kind === 'video').map((ref) => ref.name));
  const answers: BatchAnswer[] = [];

  // At least one request: an empty selection is still answered by the server.
  for (let index = 0; index < Math.max(1, imageBatches.length, videoBatches.length); index += 1) {
    answers.push(await ask({ image_names: imageBatches[index] ?? [], video_names: videoBatches[index] ?? [] }));
  }

  return answers.length === 1 ? answers[0]!.usage : mergeGalleryMediaUsage(answers);
};
