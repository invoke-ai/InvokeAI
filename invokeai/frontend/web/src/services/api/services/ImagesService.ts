/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_image } from '../models/Body_upload_image';
import type { ImageType } from '../models/ImageType';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ImagesService {

  /**
   * Get Image
   * Gets a result
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
   * Get Thumbnail
   * Gets a thumbnail
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getThumbnail({
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
      url: '/api/v1/images/{image_type}/thumbnails/{image_name}',
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
   * Upload Image
   * @returns any Successful Response
   * @throws ApiError
   */
  public static uploadImage({
    formData,
  }: {
    formData: Body_upload_image,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/images/uploads/',
      formData: formData,
      mediaType: 'multipart/form-data',
      errors: {
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

}
