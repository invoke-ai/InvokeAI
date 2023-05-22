/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ImageCategory } from '../models/ImageCategory';
import type { ImageType } from '../models/ImageType';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class RecordsService {

  /**
   * Get Image Record
   * Gets an image record by id
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getImageRecord({
    imageType,
    imageName,
  }: {
    /**
     * The type of the image record to get
     */
    imageType: ImageType,
    /**
     * The id of the image record to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/records/{image_type}/{image_name}',
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
   * List Image Records
   * Gets a list of image records by type and category
   * @returns any Successful Response
   * @throws ApiError
   */
  public static listImageRecords({
    imageType,
    imageCategory,
    page,
    perPage = 10,
  }: {
    /**
     * The type of image records to get
     */
    imageType: ImageType,
    /**
     * The kind of image records to get
     */
    imageCategory: ImageCategory,
    /**
     * The page of image records to get
     */
    page?: number,
    /**
     * The number of image records per page
     */
    perPage?: number,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/records/',
      query: {
        'image_type': imageType,
        'image_category': imageCategory,
        'page': page,
        'per_page': perPage,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
