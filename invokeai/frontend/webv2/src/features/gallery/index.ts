export type {
  GalleryBoard,
  GalleryBoardDeletionResult,
  GalleryBoardKind,
  GalleryBoardOrderBy,
  GalleryImage,
  GalleryImageMetadata,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
  GeneratedImageContract,
} from './core/types';
export {
  assertNeverGalleryItem,
  compareGalleryItems,
  formatGalleryVideoDuration,
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  toGalleryItemRef,
  type GalleryImageItem,
  type GalleryItem,
  type GalleryItemCategory,
  type GalleryItemKey,
  type GalleryItemKind,
  type GalleryItemMutationResult,
  type GalleryItemRef,
  type GalleryItemsPage,
  type GalleryVideoItem,
} from './core/items';
import {
  toGalleryItemKey as getGalleryItemKey,
  type GalleryItemKind as ItemKind,
  type GalleryItemMutationResult as ItemMutationResult,
  type GalleryItemRef as ItemRef,
} from './core/items';
import {
  addGalleryImageItemsToBoard,
  addImagesToGalleryBoard,
  deleteGalleryImageItems,
  deleteGalleryImages,
  deleteGalleryVideoItems,
  downloadGalleryArchive,
  getGalleryImageByName,
  getGalleryImageMetadata,
  getGalleryImagesByNames,
  getGalleryItemByRef,
  getGalleryVideoMetadata,
  getGalleryVideoWorkflow,
  isDateBoardId,
  listGalleryBoards,
  makeImageCanvasAsset,
  makeImageDurable,
  moveGalleryVideoItemsToBoard,
  removeGalleryImageItemsFromBoard,
  removeImagesFromGalleryBoard,
  saveImageToGallery,
  setGalleryImageItemsStarred,
  setGalleryVideoItemsStarred,
  starGalleryImages,
  type GalleryItemOrganizationTransportResult,
  unstarGalleryImages,
  uploadGalleryImage,
} from './data/backend';

/** Resolve backend Gallery images without exposing transport DTOs or endpoints. */
export const galleryImages = {
  metadata: getGalleryImageMetadata,
  resolve: getGalleryImageByName,
  resolveMany: getGalleryImagesByNames,
} as const;

/** Resolve either media kind while keeping the legacy image port strictly image-only. */
export const galleryItems = {
  resolve: getGalleryItemByRef,
} as const;

/** Focused read-only details needed by mixed-media preview surfaces. */
export const galleryVideos = {
  metadata: getGalleryVideoMetadata,
  workflow: getGalleryVideoWorkflow,
} as const;

/** Import/export intents shared by image-producing and image-consuming features. */
export const galleryTransfers = {
  downloadArchive: downloadGalleryArchive,
  upload: uploadGalleryImage,
} as const;

/** Durability transitions for intermediate results. */
export const galleryDurability = {
  /** Durable, and adopted as canvas-owned pixels (out of the gallery's Images view). */
  makeCanvasAsset: makeImageCanvasAsset,
  makeDurable: makeImageDurable,
  save: saveImageToGallery,
} as const;

/** Board and image organization intents. */
export const galleryOrganization = {
  addToBoard: addImagesToGalleryBoard,
  deleteImages: deleteGalleryImages,
  removeFromBoard: removeImagesFromGalleryBoard,
  setStarred: (imageNames: string[], starred: boolean, signal?: AbortSignal): Promise<string[]> =>
    starred ? starGalleryImages(imageNames, signal) : unstarGalleryImages(imageNames, signal),
} as const;

type ConfirmedGalleryItemMutationResult = ItemMutationResult & { affectedBoardIds: string[] };

type GalleryItemPartitionMutation = (
  names: string[],
  signal?: AbortSignal
) => Promise<GalleryItemOrganizationTransportResult>;

const emptyGalleryItemTransportResult = (): GalleryItemOrganizationTransportResult => ({
  affectedBoardIds: [],
  succeededNames: [],
});

const deduplicateGalleryItemRefs = (refs: readonly ItemRef[]): ItemRef[] => {
  const keys = new Set<string>();
  const unique: ItemRef[] = [];

  for (const ref of refs) {
    const key = getGalleryItemKey(ref);

    if (keys.has(key)) {
      continue;
    }
    keys.add(key);
    unique.push(ref);
  }

  return unique;
};

const addConfirmedPartition = ({
  affectedBoardIds,
  confirmedKeys,
  kind,
  names,
  outcome,
}: {
  affectedBoardIds: Set<string>;
  confirmedKeys: Set<string>;
  kind: ItemKind;
  names: string[];
  outcome: PromiseSettledResult<GalleryItemOrganizationTransportResult>;
}): void => {
  if (outcome.status === 'rejected') {
    return;
  }

  const requestedNames = new Set(names);
  let hasConfirmedSuccess = false;

  for (const name of outcome.value.succeededNames) {
    if (!requestedNames.has(name)) {
      continue;
    }
    confirmedKeys.add(getGalleryItemKey({ kind, name }));
    hasConfirmedSuccess = true;
  }

  if (!hasConfirmedSuccess) {
    return;
  }
  for (const boardId of outcome.value.affectedBoardIds) {
    affectedBoardIds.add(boardId);
  }
};

const mutateGalleryItems = async (
  refs: readonly ItemRef[],
  imageMutation: GalleryItemPartitionMutation,
  videoMutation: GalleryItemPartitionMutation,
  signal?: AbortSignal
): Promise<ConfirmedGalleryItemMutationResult> => {
  const requested = deduplicateGalleryItemRefs(refs);
  const imageNames = requested.filter((ref) => ref.kind === 'image').map((ref) => ref.name);
  const videoNames = requested.filter((ref) => ref.kind === 'video').map((ref) => ref.name);
  const [imageOutcome, videoOutcome] = await Promise.allSettled([
    imageNames.length > 0 ? imageMutation(imageNames, signal) : Promise.resolve(emptyGalleryItemTransportResult()),
    videoNames.length > 0 ? videoMutation(videoNames, signal) : Promise.resolve(emptyGalleryItemTransportResult()),
  ]);
  const affectedBoardIds = new Set<string>();
  const confirmedKeys = new Set<string>();

  addConfirmedPartition({
    affectedBoardIds,
    confirmedKeys,
    kind: 'image',
    names: imageNames,
    outcome: imageOutcome,
  });
  addConfirmedPartition({
    affectedBoardIds,
    confirmedKeys,
    kind: 'video',
    names: videoNames,
    outcome: videoOutcome,
  });

  return {
    affectedBoardIds: [...affectedBoardIds].sort(),
    failed: requested.filter((ref) => !confirmedKeys.has(getGalleryItemKey(ref))),
    succeeded: requested.filter((ref) => confirmedKeys.has(getGalleryItemKey(ref))),
  };
};

/**
 * Mixed-media organization transport. Results contain only backend-confirmed
 * refs and do not patch caches, invalidate queries, or notify.
 */
export const galleryItemOrganization = {
  delete: (refs: readonly ItemRef[], signal?: AbortSignal): Promise<ConfirmedGalleryItemMutationResult> =>
    mutateGalleryItems(refs, deleteGalleryImageItems, deleteGalleryVideoItems, signal),
  moveToBoard: (
    refs: readonly ItemRef[],
    boardId: string,
    signal?: AbortSignal
  ): Promise<ConfirmedGalleryItemMutationResult> =>
    mutateGalleryItems(
      refs,
      (names, requestSignal) =>
        boardId === 'none'
          ? removeGalleryImageItemsFromBoard(names, requestSignal)
          : addGalleryImageItemsToBoard(boardId, names, requestSignal),
      (names, requestSignal) => moveGalleryVideoItemsToBoard(names, boardId, requestSignal),
      signal
    ),
  setStarred: (
    refs: readonly ItemRef[],
    starred: boolean,
    signal?: AbortSignal
  ): Promise<ConfirmedGalleryItemMutationResult> =>
    mutateGalleryItems(
      refs,
      (names, requestSignal) => setGalleryImageItemsStarred(names, starred, requestSignal),
      (names, requestSignal) => setGalleryVideoItemsStarred(names, starred, requestSignal),
      signal
    ),
} as const;

/** Destination choices for callers such as Workflow fields. */
export const galleryDestinations = {
  list: listGalleryBoards,
} as const;

export const isGalleryVirtualBoard = isDateBoardId;
