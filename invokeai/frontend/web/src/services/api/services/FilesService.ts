/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ImageType } from '../models/ImageType';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class FilesService {

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
     * The type of the image to get
     */
    imageType: ImageType,
    /**
     * The id of the image to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/files/images/{image_type}/{image_name}',
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
     * The type of the image whose thumbnail to get
     */
    imageType: ImageType,
    /**
     * The id of the image whose thumbnail to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/files/images/{image_type}/{image_name}/thumbnail',
      path: {
        'image_type': imageType,
        'image_name': imageName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
