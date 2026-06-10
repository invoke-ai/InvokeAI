import type { GalleryBoard, GalleryImage, GalleryView } from '../../gallery/api';
import type { GeneratedImageContract } from '../../types';

const UNCATEGORIZED_BOARD: GalleryBoard = {
  assetCount: 0,
  id: 'none',
  imageCount: 0,
  isVirtual: true,
  name: 'Uncategorized',
};

export interface GalleryStateView {
  boards: GalleryBoard[];
  galleryView: GalleryView;
  imageDensityPercent: number;
  images: GalleryImage[];
  isLoading: boolean;
  searchTerm: string;
  selectedBoardId: string;
  selectedImageName: string | null;
}

export const getGalleryView = (values: Record<string, unknown>): GalleryView =>
  values.galleryView === 'assets' ? 'assets' : 'images';

export const getGallerySearchTerm = (values: Record<string, unknown>): string =>
  typeof values.searchTerm === 'string' ? values.searchTerm : '';

export const getGallerySelectedBoardId = (values: Record<string, unknown>, backendBoards: GalleryBoard[]): string => {
  const selectedBoardId = typeof values.selectedBoardId === 'string' ? values.selectedBoardId : 'none';

  if (backendBoards.length === 0 || backendBoards.some((board) => board.id === selectedBoardId)) {
    return selectedBoardId;
  }

  return 'none';
};

export const getGalleryRecentImagesKey = (values: Record<string, unknown>): string => {
  if (!Array.isArray(values.recentImages)) {
    return '';
  }

  return (values.recentImages as GeneratedImageContract[]).map((image) => image.imageName).join('\0');
};

export const getGalleryStateView = (
  values: Record<string, unknown>,
  backendBoards: GalleryBoard[],
  backendImages: GalleryImage[] | null,
  isLoading: boolean
): GalleryStateView => {
  const localImages = Array.isArray(values.recentImages)
    ? (values.recentImages as GeneratedImageContract[]).map((image) => ({
        ...image,
        boardId: 'none',
        imageCategory: 'general' as const,
      }))
    : [];
  const images = backendImages ?? (isLoading ? [] : localImages);
  const selectedImageName = typeof values.selectedImageName === 'string' ? values.selectedImageName : null;
  const galleryView = getGalleryView(values);
  const imageDensityPercent =
    typeof values.imageDensityPercent === 'number' && Number.isFinite(values.imageDensityPercent)
      ? Math.min(100, Math.max(0, values.imageDensityPercent))
      : 50;
  const searchTerm = getGallerySearchTerm(values);
  const boards = backendBoards.length ? backendBoards : [{ ...UNCATEGORIZED_BOARD, imageCount: images.length }];
  const selectedBoardId = getGallerySelectedBoardId(values, backendBoards);

  return {
    boards,
    galleryView,
    imageDensityPercent,
    images,
    isLoading,
    searchTerm,
    selectedBoardId,
    selectedImageName: images.some((image) => image.imageName === selectedImageName) ? selectedImageName : null,
  };
};

export const getBoardCounts = (board: GalleryBoard): { assetCount: number; imageCount: number } => ({
  assetCount: board.assetCount,
  imageCount: board.imageCount,
});
