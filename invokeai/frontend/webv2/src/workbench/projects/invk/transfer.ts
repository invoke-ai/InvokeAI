import type { ProjectAssetRefs } from '@workbench/projects/projectAssets';

import type { InvkBoardItem, InvkMediaKind } from './board';

/**
 * The vocabulary shared by everything that moves a project's media: export, import, and
 * duplication.
 *
 * All three face the same problem. A project's media comes from two places that overlap — the
 * board, which is membership, and the document, which is references — and an item can be in
 * either, or both. What must happen to it differs by which:
 *
 * - **Board membership** must be *copied*. `board_images` keys on the image name, so one image sits
 *   on exactly one board; a restored project that reused an existing name would be sharing another
 *   project's media, and deleting either board would take it from both.
 * - **Document references** may be *reused*. They are pointers, not membership, so an image the
 *   destination already has satisfies the reference without a second copy. This is the v2
 *   behaviour, and it is still right for media that lives outside the project's own board.
 *
 * The consequence is the rule this module exists to encode: an item that is *both* is restored as
 * board media, and its document references are rewritten to the copy. And when that copy fails, the
 * reference must not be left pointing at the old name — see {@link buildMissingMediaName}.
 */

/** One item, in whichever namespace it belongs to. */
export interface InvkMediaRef {
  kind: InvkMediaKind;
  name: string;
}

/** Stable identity across both namespaces. An image and a video may share a name. */
export const toMediaKey = ({ kind, name }: InvkMediaRef): string => `${kind}:${name}`;

export const compareMediaRefs = (left: InvkMediaRef, right: InvkMediaRef): number =>
  left.kind.localeCompare(right.kind) || left.name.localeCompare(right.name);

/** Why an item did not make it. Each is reported, never thrown: a partial project beats none. */
export type InvkMediaIssueReason =
  /** The exporting server would not serve the bytes. */
  | 'fetch-failed'
  /** The archive named the item on its board but carried no bytes for it. */
  | 'missing-entry'
  /** The destination server rejected the upload or copy. */
  | 'upload-failed'
  /** The media arrived, but could not be re-starred. */
  | 'star-failed';

export interface InvkMediaIssue extends InvkMediaRef {
  reason: InvkMediaIssueReason;
}

/**
 * What a transfer could not carry, split by what it means to the person holding the file.
 *
 * A missing board item costs a result they can still see elsewhere. A missing document reference
 * costs a layer in the canvas — the project opens with a hole in it. Reporting one number for both
 * made "12 assets could not be included" mean anything from "nothing you will notice" to "the
 * canvas is broken", so they are counted apart.
 *
 * An item that was both appears in both arrays. That is not double-counting: it genuinely failed
 * in both roles.
 */
export interface ProjectTransferIssues {
  boardItemIssues: InvkMediaIssue[];
  documentReferenceIssues: InvkMediaIssue[];
}

const compareIssues = (left: InvkMediaIssue, right: InvkMediaIssue): number =>
  compareMediaRefs(left, right) || left.reason.localeCompare(right.reason);

/** Collects issues in whatever order they happen and hands them back in a stable one. */
export const createTransferIssueLog = (): {
  addBoardItemIssue: (ref: InvkMediaRef, reason: InvkMediaIssueReason) => void;
  addDocumentReferenceIssue: (ref: InvkMediaRef, reason: InvkMediaIssueReason) => void;
  toIssues: () => ProjectTransferIssues;
} => {
  const boardItemIssues: InvkMediaIssue[] = [];
  const documentReferenceIssues: InvkMediaIssue[] = [];

  return {
    addBoardItemIssue: ({ kind, name }, reason) => boardItemIssues.push({ kind, name, reason }),
    addDocumentReferenceIssue: ({ kind, name }, reason) => documentReferenceIssues.push({ kind, name, reason }),
    toIssues: () => ({
      boardItemIssues: [...boardItemIssues].sort(compareIssues),
      documentReferenceIssues: [...documentReferenceIssues].sort(compareIssues),
    }),
  };
};

/** Turn the collector's per-kind sets into refs. */
export const toMediaRefs = (refs: ProjectAssetRefs): InvkMediaRef[] =>
  [
    ...[...refs.images].map((name) => ({ kind: 'image' as const, name })),
    ...[...refs.videos].map((name) => ({ kind: 'video' as const, name })),
  ].sort(compareMediaRefs);

/** One item of the union, and which side (or sides) it came from. */
export interface InvkTransferItem extends InvkMediaRef {
  /** On the project's board, so it must be copied rather than reused. */
  isBoardItem: boolean;
  /** Named by the document, so a failure leaves a hole in the canvas. */
  isDocumentReference: boolean;
  /** Present only for board items; the category and starring to restore. */
  boardItem: InvkBoardItem | null;
}

/**
 * Merge board membership and document references into one list, each item appearing once.
 *
 * Fetching is the expensive half of an export and uploading the expensive half of an import, so an
 * item that is both must be handled once, not twice — and the union is also what lets a single
 * failure be reported correctly against both roles.
 */
export const planMediaTransfer = (
  boardItems: readonly InvkBoardItem[],
  documentRefs: readonly InvkMediaRef[]
): InvkTransferItem[] => {
  const byKey = new Map<string, InvkTransferItem>();

  for (const boardItem of boardItems) {
    byKey.set(toMediaKey(boardItem), {
      boardItem,
      isBoardItem: true,
      isDocumentReference: false,
      kind: boardItem.kind,
      name: boardItem.name,
    });
  }

  for (const ref of documentRefs) {
    const key = toMediaKey(ref);
    const existing = byKey.get(key);

    if (existing) {
      existing.isDocumentReference = true;
      continue;
    }

    byKey.set(key, {
      boardItem: null,
      isBoardItem: false,
      isDocumentReference: true,
      kind: ref.kind,
      name: ref.name,
    });
  }

  return [...byKey.values()].sort(compareMediaRefs);
};

/**
 * A name guaranteed not to resolve, for a document reference whose board media could not be
 * restored.
 *
 * Leaving the original name in place would be worse than a broken reference. Names are assigned by
 * the exporting server, and the destination may well have its own image under that exact name —
 * on the same server, during a duplication, it certainly does. The project would then open
 * pointing at a stranger's picture, silently and plausibly, with nothing to indicate it is the
 * wrong one. A reference that resolves to nothing is honest: it renders as a missing layer,
 * exactly as a dangling v2 reference already does.
 *
 * Derived from the new project id, so it is stable within one import (every occurrence of the old
 * name maps to the same placeholder) and unique across imports.
 */
export const buildMissingMediaName = (projectId: string, kind: InvkMediaKind, index: number): string =>
  `${projectId}-missing-${kind}-${index}`;
