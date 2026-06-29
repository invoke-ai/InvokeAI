import type { GalleryBoardKind } from '@workbench/gallery/api';

export interface GalleryImageDragImage {
  boardId: string;
  imageName: string;
}

export interface GalleryImageDragData {
  kind: 'gallery-image';
  images: GalleryImageDragImage[];
}

export interface GalleryBoardDropData {
  boardId: string;
  boardKind: GalleryBoardKind;
  kind: 'gallery-board';
}

export const getGalleryImageDragId = (imageName: string): string => `gallery-image:${imageName}`;

export const getGalleryBoardDropId = (boardId: string): string => `gallery-board:${boardId}`;

export const getGalleryImageDragData = (images: GalleryImageDragImage[]): GalleryImageDragData => ({
  images,
  kind: 'gallery-image',
});

export const getGalleryBoardDropData = (boardId: string, boardKind: GalleryBoardKind): GalleryBoardDropData => ({
  boardId,
  boardKind,
  kind: 'gallery-board',
});

export const isGalleryImageDragData = (value: unknown): value is GalleryImageDragData =>
  isRecord(value) &&
  value.kind === 'gallery-image' &&
  Array.isArray(value.images) &&
  value.images.every(
    (image) => isRecord(image) && typeof image.imageName === 'string' && typeof image.boardId === 'string'
  );

export const isGalleryBoardDropData = (value: unknown): value is GalleryBoardDropData =>
  isRecord(value) &&
  value.kind === 'gallery-board' &&
  typeof value.boardId === 'string' &&
  isGalleryBoardKind(value.boardKind);

export const getGalleryImageNamesOutsideBoard = (dragData: GalleryImageDragData, boardId: string): string[] =>
  dragData.images.filter((image) => image.boardId !== boardId).map((image) => image.imageName);

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isGalleryBoardKind = (value: unknown): value is GalleryBoardKind =>
  value === 'board' || value === 'date' || value === 'uncategorized';
