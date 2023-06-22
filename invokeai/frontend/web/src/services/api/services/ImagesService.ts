/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_image } from '../models/Body_upload_image';
import type { ImageCategory } from '../models/ImageCategory';
import type { ImageDTO } from '../models/ImageDTO';
import type { ImageRecordChanges } from '../models/ImageRecordChanges';
import type { ImageUrlsDTO } from '../models/ImageUrlsDTO';
import type { OffsetPaginatedResults_ImageDTO_ } from '../models/OffsetPaginatedResults_ImageDTO_';
import type { ResourceOrigin } from '../models/ResourceOrigin';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ImagesService {

  /**
   * List Images With Metadata
   * Gets a list of images
   * @returns OffsetPaginatedResults_ImageDTO_ Successful Response
   * @throws ApiError
   */
  public static listImagesWithMetadata({
    imageOrigin,
    categories,
    isIntermediate,
    boardId,
    offset,
    limit = 10,
  }: {
    /**
     * The origin of images to list
     */
    imageOrigin?: ResourceOrigin,
    /**
     * The categories of image to include
     */
    categories?: Array<ImageCategory>,
    /**
     * Whether to list intermediate images
     */
    isIntermediate?: boolean,
    /**
     * The board id to filter by
     */
    boardId?: string,
    /**
     * The page offset
     */
    offset?: number,
    /**
     * The number of images per page
     */
    limit?: number,
  }): CancelablePromise<OffsetPaginatedResults_ImageDTO_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/',
      query: {
        'image_origin': imageOrigin,
        'categories': categories,
        'is_intermediate': isIntermediate,
        'board_id': boardId,
        'offset': offset,
        'limit': limit,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Upload Image
   * Uploads an image
   * @returns ImageDTO The image was uploaded successfully
   * @throws ApiError
   */
  public static uploadImage({
    imageCategory,
    isIntermediate,
    formData,
    sessionId,
  }: {
    /**
     * The category of the image
     */
    imageCategory: ImageCategory,
    /**
     * Whether this is an intermediate image
     */
    isIntermediate: boolean,
    formData: Body_upload_image,
    /**
     * The session ID associated with this upload, if any
     */
    sessionId?: string,
  }): CancelablePromise<ImageDTO> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/images/',
      query: {
        'image_category': imageCategory,
        'is_intermediate': isIntermediate,
        'session_id': sessionId,
      },
      formData: formData,
      mediaType: 'multipart/form-data',
      errors: {
        415: `Image upload failed`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Image Full
   * Gets a full-resolution image file
   * @returns any Return the full-resolution image
   * @throws ApiError
   */
  public static getImageFull({
    imageName,
  }: {
    /**
     * The name of full-resolution image file to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_name}',
      path: {
        'image_name': imageName,
      },
      errors: {
        404: `Image not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Image
   * Deletes an image
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteImage({
    imageName,
  }: {
    /**
     * The name of the image to delete
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/images/{image_name}',
      path: {
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Update Image
   * Updates an image
   * @returns ImageDTO Successful Response
   * @throws ApiError
   */
  public static updateImage({
    imageName,
    requestBody,
  }: {
    /**
     * The name of the image to update
     */
    imageName: string,
    requestBody: ImageRecordChanges,
  }): CancelablePromise<ImageDTO> {
    return __request(OpenAPI, {
      method: 'PATCH',
      url: '/api/v1/images/{image_name}',
      path: {
        'image_name': imageName,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Image Metadata
   * Gets an image's metadata
   * @returns ImageDTO Successful Response
   * @throws ApiError
   */
  public static getImageMetadata({
    imageName,
  }: {
    /**
     * The name of image to get
     */
    imageName: string,
  }): CancelablePromise<ImageDTO> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_name}/metadata',
      path: {
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Image Thumbnail
   * Gets a thumbnail image file
   * @returns any Return the image thumbnail
   * @throws ApiError
   */
  public static getImageThumbnail({
    imageName,
  }: {
    /**
     * The name of thumbnail image file to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_name}/thumbnail',
      path: {
        'image_name': imageName,
      },
      errors: {
        404: `Image not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Image Urls
   * Gets an image and thumbnail URL
   * @returns ImageUrlsDTO Successful Response
   * @throws ApiError
   */
  public static getImageUrls({
    imageName,
  }: {
    /**
     * The name of the image whose URL to get
     */
    imageName: string,
  }): CancelablePromise<ImageUrlsDTO> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_name}/urls',
      path: {
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
