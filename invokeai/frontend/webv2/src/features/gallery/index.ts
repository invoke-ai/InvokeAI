export type {
  GalleryBoard,
  GalleryBoardKind,
  GalleryBoardOrderBy,
  GalleryImage,
  GalleryImageMetadata,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
  GeneratedImageContract,
} from './core/types';
import {
  addImagesToGalleryBoard,
  deleteGalleryImages,
  downloadGalleryArchive,
  getGalleryImageByName,
  getGalleryImageMetadata,
  getGalleryImagesByNames,
  getImageFullUrl,
  getImageThumbnailUrl,
  isDateBoardId,
  listGalleryBoards,
  makeImageDurable,
  removeImagesFromGalleryBoard,
  saveImageToGallery,
  starGalleryImages,
  unstarGalleryImages,
  uploadGalleryImage,
} from './data/backend';

/** Resolve backend Gallery images without exposing transport DTOs or endpoints. */
export const galleryImages = {
  fullUrl: getImageFullUrl,
  metadata: getGalleryImageMetadata,
  resolve: getGalleryImageByName,
  resolveMany: getGalleryImagesByNames,
  thumbnailUrl: getImageThumbnailUrl,
} as const;

/** Import/export intents shared by image-producing and image-consuming features. */
export const galleryTransfers = {
  downloadArchive: downloadGalleryArchive,
  upload: uploadGalleryImage,
} as const;

/** Durability transitions for intermediate results. */
export const galleryDurability = {
  makeDurable: makeImageDurable,
  save: saveImageToGallery,
} as const;

/** Board and image organization intents. */
export const galleryOrganization = {
  addToBoard: addImagesToGalleryBoard,
  deleteImages: deleteGalleryImages,
  removeFromBoard: removeImagesFromGalleryBoard,
  setStarred: (imageNames: string[], starred: boolean): Promise<void> =>
    starred ? starGalleryImages(imageNames) : unstarGalleryImages(imageNames),
} as const;

/** Destination choices for callers such as Workflow fields. */
export const galleryDestinations = {
  list: listGalleryBoards,
} as const;

export const isGalleryVirtualBoard = isDateBoardId;
