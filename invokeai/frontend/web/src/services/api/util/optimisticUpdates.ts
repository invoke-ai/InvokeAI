import type { ImageDTO, ImageNamesResult } from 'services/api/types';

/**
 * Calculates the optimal insertion position for a new image in the names list.
 * For starred_first=true: starred images go to position 0, unstarred go after all starred images
 * For starred_first=false: all new images go to position 0 (newest first)
 */
export function calculateImageInsertionPosition(
  imageDTO: ImageDTO,
  starredFirst: boolean,
  starredCount: number
): number {
  if (!starredFirst) {
    // When starred_first is false, always insert at the beginning (newest first)
    return 0;
  }

  // When starred_first is true
  if (imageDTO.starred) {
    // Starred images go at the very beginning
    return 0;
  }

  // Unstarred images go after all starred images
  return starredCount;
}

/**
 * Optimistically inserts a new image into the ImageNamesResult at the correct position
 */
export function insertImageIntoNamesResult(
  currentResult: ImageNamesResult,
  imageDTO: ImageDTO,
  starredFirst: boolean
): ImageNamesResult {
  // Don't insert if the image is already in the list
  if (currentResult.image_names.includes(imageDTO.image_name)) {
    return currentResult;
  }

  const insertPosition = calculateImageInsertionPosition(imageDTO, starredFirst, currentResult.starred_count);

  const newImageNames = [...currentResult.image_names];
  newImageNames.splice(insertPosition, 0, imageDTO.image_name);

  return {
    image_names: newImageNames,
    starred_count: starredFirst && imageDTO.starred ? currentResult.starred_count + 1 : currentResult.starred_count,
    total_count: currentResult.total_count + 1,
  };
}

/**
 * Optimistically removes an image from the ImageNamesResult
 */
export function removeImageFromNamesResult(
  currentResult: ImageNamesResult,
  imageNameToRemove: string,
  wasStarred: boolean,
  starredFirst: boolean
): ImageNamesResult {
  const newImageNames = currentResult.image_names.filter((name) => name !== imageNameToRemove);

  return {
    image_names: newImageNames,
    starred_count: starredFirst && wasStarred ? currentResult.starred_count - 1 : currentResult.starred_count,
    total_count: currentResult.total_count - 1,
  };
}

/**
 * Optimistically updates an image's position in the result when its starred status changes
 */
export function updateImagePositionInNamesResult(
  currentResult: ImageNamesResult,
  updatedImageDTO: ImageDTO,
  previouslyStarred: boolean,
  starredFirst: boolean
): ImageNamesResult {
  // First remove the image from its current position
  const withoutImage = removeImageFromNamesResult(
    currentResult,
    updatedImageDTO.image_name,
    previouslyStarred,
    starredFirst
  );

  // Then insert it at the new correct position
  return insertImageIntoNamesResult(withoutImage, updatedImageDTO, starredFirst);
}
