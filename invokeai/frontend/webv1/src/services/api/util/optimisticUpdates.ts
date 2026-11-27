import type { OrderDir } from 'features/gallery/store/types';
import type { GetImageNamesResult, ImageDTO } from 'services/api/types';

/**
 * Calculates the optimal insertion position for a new image in the names list.
 * For starred_first=true: starred images go to position 0, unstarred go after all starred images
 * For starred_first=false: all new images go to position 0 (newest first)
 */
function calculateImageInsertionPosition(
  imageDTO: ImageDTO,
  starredFirst: boolean,
  starredCount: number,
  orderDir: OrderDir = 'DESC'
): number {
  if (!starredFirst) {
    // When starred_first is false, insertion depends on order direction
    return orderDir === 'DESC' ? 0 : Number.MAX_SAFE_INTEGER;
  }

  // When starred_first is true
  if (imageDTO.starred) {
    // Starred images: beginning for desc, after existing starred for asc
    return orderDir === 'DESC' ? 0 : starredCount;
  }

  // Unstarred images go after all starred images
  return orderDir === 'DESC' ? starredCount : Number.MAX_SAFE_INTEGER;
}

/**
 * Optimistically inserts a new image into the ImageNamesResult at the correct position
 */
export function insertImageIntoNamesResult(
  currentResult: GetImageNamesResult,
  imageDTO: ImageDTO,
  starredFirst: boolean,
  orderDir: OrderDir = 'DESC'
): GetImageNamesResult {
  // Don't insert if the image is already in the list
  if (currentResult.image_names.includes(imageDTO.image_name)) {
    return currentResult;
  }

  const insertPosition = calculateImageInsertionPosition(imageDTO, starredFirst, currentResult.starred_count, orderDir);

  const newImageNames = [...currentResult.image_names];
  // Handle MAX_SAFE_INTEGER by pushing to end
  if (insertPosition >= newImageNames.length) {
    newImageNames.push(imageDTO.image_name);
  } else {
    newImageNames.splice(insertPosition, 0, imageDTO.image_name);
  }

  return {
    image_names: newImageNames,
    starred_count: starredFirst && imageDTO.starred ? currentResult.starred_count + 1 : currentResult.starred_count,
    total_count: currentResult.total_count + 1,
  };
}
