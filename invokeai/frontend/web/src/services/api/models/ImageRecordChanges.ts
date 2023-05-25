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
};

