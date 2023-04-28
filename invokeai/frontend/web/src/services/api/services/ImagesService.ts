/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_image } from '../models/Body_upload_image';
import type { ImageResponse } from '../models/ImageResponse';
import type { ImageType } from '../models/ImageType';
import type { PaginatedResults_ImageResponse_ } from '../models/PaginatedResults_ImageResponse_';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ImagesService {

  /**
   * Get Image
   * Gets an image
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getImage({
    imageType,
    imageName,
  }: {
    /**
     * The type of image to get
     */
    imageType: ImageType,
    /**
     * The name of the image to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_type}/{image_name}',
      path: {
        'image_type': imageType,
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Image
   * Deletes an image and its thumbnail
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteImage({
    imageType,
    imageName,
  }: {
    /**
     * The type of image to delete
     */
    imageType: ImageType,
    /**
     * The name of the image to delete
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/images/{image_type}/{image_name}',
      path: {
        'image_type': imageType,
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Thumbnail
   * Gets a thumbnail
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getThumbnail({
    thumbnailType,
    thumbnailName,
  }: {
    /**
     * The type of thumbnail to get
     */
    thumbnailType: ImageType,
    /**
     * The name of the thumbnail to get
     */
    thumbnailName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{thumbnail_type}/thumbnails/{thumbnail_name}',
      path: {
        'thumbnail_type': thumbnailType,
        'thumbnail_name': thumbnailName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Upload Image
   * @returns ImageResponse The image was uploaded successfully
   * @throws ApiError
   */
  public static uploadImage({
    formData,
  }: {
    formData: Body_upload_image,
  }): CancelablePromise<ImageResponse> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/images/uploads/',
      formData: formData,
      mediaType: 'multipart/form-data',
      errors: {
        415: `Image upload failed`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * List Images
   * Gets a list of images
   * @returns PaginatedResults_ImageResponse_ Successful Response
   * @throws ApiError
   */
  public static listImages({
    imageType,
    page,
    perPage = 10,
  }: {
    /**
     * The type of images to get
     */
    imageType?: ImageType,
    /**
     * The page of images to get
     */
    page?: number,
    /**
     * The number of images per page
     */
    perPage?: number,
  }): CancelablePromise<PaginatedResults_ImageResponse_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/',
      query: {
        'image_type': imageType,
        'page': page,
        'per_page': perPage,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
