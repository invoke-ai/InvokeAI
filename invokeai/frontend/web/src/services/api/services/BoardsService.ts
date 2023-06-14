/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { BoardChanges } from '../models/BoardChanges';
import type { Body_create_board_image } from '../models/Body_create_board_image';
import type { Body_remove_board_image } from '../models/Body_remove_board_image';
import type { OffsetPaginatedResults_BoardRecord_ } from '../models/OffsetPaginatedResults_BoardRecord_';
import type { OffsetPaginatedResults_ImageDTO_ } from '../models/OffsetPaginatedResults_ImageDTO_';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class BoardsService {

  /**
   * List Boards
   * Gets a list of boards
   * @returns OffsetPaginatedResults_BoardRecord_ Successful Response
   * @throws ApiError
   */
  public static listBoards({
    offset,
    limit = 10,
  }: {
    /**
     * The page offset
     */
    offset?: number,
    /**
     * The number of boards per page
     */
    limit?: number,
  }): CancelablePromise<OffsetPaginatedResults_BoardRecord_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/boards/',
      query: {
        'offset': offset,
        'limit': limit,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Create Board
   * Creates a board
   * @returns any The board was created successfully
   * @throws ApiError
   */
  public static createBoard({
    requestBody,
  }: {
    requestBody: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/boards/',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Board
   * Deletes a board
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteBoard({
    boardId,
  }: {
    /**
     * The id of board to delete
     */
    boardId: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/boards/{board_id}',
      path: {
        'board_id': boardId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Update Board
   * Creates a board
   * @returns any The board was updated successfully
   * @throws ApiError
   */
  public static updateBoard({
    boardId,
    requestBody,
  }: {
    /**
     * The id of board to update
     */
    boardId: string,
    requestBody: BoardChanges,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'PATCH',
      url: '/api/v1/boards/{board_id}',
      path: {
        'board_id': boardId,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Create Board Image
   * Creates a board_image
   * @returns any The image was added to a board successfully
   * @throws ApiError
   */
  public static createBoardImage({
    requestBody,
  }: {
    requestBody: Body_create_board_image,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/board_images/',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Remove Board Image
   * Deletes a board_image
   * @returns any The image was removed from the board successfully
   * @throws ApiError
   */
  public static removeBoardImage({
    requestBody,
  }: {
    requestBody: Body_remove_board_image,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/board_images/',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * List Board Images
   * Gets a list of images for a board
   * @returns OffsetPaginatedResults_ImageDTO_ Successful Response
   * @throws ApiError
   */
  public static listBoardImages({
    boardId,
    offset,
    limit = 10,
  }: {
    /**
     * The id of the board
     */
    boardId: string,
    /**
     * The page offset
     */
    offset?: number,
    /**
     * The number of boards per page
     */
    limit?: number,
  }): CancelablePromise<OffsetPaginatedResults_ImageDTO_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/board_images/{board_id}',
      path: {
        'board_id': boardId,
      },
      query: {
        'offset': offset,
        'limit': limit,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
