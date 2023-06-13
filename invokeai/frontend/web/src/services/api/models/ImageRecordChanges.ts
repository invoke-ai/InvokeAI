/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ImageCategory } from './ImageCategory';

/**
 * A set of changes to apply to an image record.
 *
 * Only limited changes are valid:
 * - `image_category`: change the category of an image
 * - `session_id`: change the session associated with an image
 * - `is_intermediate`: change the image's `is_intermediate` flag
 */
export type ImageRecordChanges = {
  /**
   * The image's new category.
   */
  image_category?: ImageCategory;
  /**
   * The image's new session ID.
   */
  session_id?: string;
  /**
   * The image's new `is_intermediate` flag.
   */
  is_intermediate?: boolean;
};

