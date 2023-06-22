/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Deserialized board record with cover image URL and image count.
 */
export type BoardDTO = {
  /**
   * The unique ID of the board.
   */
  board_id: string;
  /**
   * The name of the board.
   */
  board_name: string;
  /**
   * The created timestamp of the board.
   */
  created_at: string;
  /**
   * The updated timestamp of the board.
   */
  updated_at: string;
  /**
   * The deleted timestamp of the board.
   */
  deleted_at?: string;
  /**
   * The name of the board's cover image.
   */
  cover_image_name?: string;
  /**
   * The number of images in the board.
   */
  image_count: number;
};

