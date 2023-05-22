/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_image } from '../models/Body_upload_image';
import type { ImageCategory } from '../models/ImageCategory';
import type { ImageType } from '../models/ImageType';

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

  /**
   * Upload Image
   * Uploads an image
   * @returns any The image was uploaded successfully
   * @throws ApiError
   */
  public static uploadImage({
    imageType,
    formData,
    imageCategory,
  }: {
    imageType: ImageType,
    formData: Body_upload_image,
    imageCategory?: ImageCategory,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/images/',
      query: {
        'image_type': imageType,
        'image_category': imageCategory,
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
   * Delete Image
   * Deletes an image
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteImage({
    imageType,
    imageName,
  }: {
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

}
