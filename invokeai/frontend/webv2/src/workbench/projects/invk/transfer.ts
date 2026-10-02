import type { ProjectAssetRefs } from '@workbench/projects/projectAssets';

import type { InvkBoardItem, InvkMediaKind } from './board';

/**
 * Board media needs fresh ownership; document-only references may be reused. Shared references remap to the copy,
 * and failed copies receive missing placeholders.
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

/** Report losses separately by role; an item may fail in both. */
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

/** Materialize once while retaining both roles for loss reporting. */
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

/** Missing names are stable within an import and cannot resolve to existing source media. */
export const buildMissingMediaName = (projectId: string, kind: InvkMediaKind, index: number): string =>
  `${projectId}-missing-${kind}-${index}`;
