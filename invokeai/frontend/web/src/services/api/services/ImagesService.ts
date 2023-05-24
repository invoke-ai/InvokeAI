/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_image } from '../models/Body_upload_image';
import type { ImageCategory } from '../models/ImageCategory';
import type { ImageDTO } from '../models/ImageDTO';
import type { ImageType } from '../models/ImageType';
import type { ImageUrlsDTO } from '../models/ImageUrlsDTO';
import type { PaginatedResults_ImageDTO_ } from '../models/PaginatedResults_ImageDTO_';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ImagesService {

  /**
   * List Images With Metadata
   * Gets a list of images with metadata
   * @returns PaginatedResults_ImageDTO_ Successful Response
   * @throws ApiError
   */
  public static listImagesWithMetadata({
    imageType,
    imageCategory,
    page,
    perPage = 10,
  }: {
    /**
     * The type of images to list
     */
    imageType: ImageType,
    /**
     * The kind of images to list
     */
    imageCategory: ImageCategory,
    /**
     * The page of image metadata to get
     */
    page?: number,
    /**
     * The number of image metadata per page
     */
    perPage?: number,
  }): CancelablePromise<PaginatedResults_ImageDTO_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/',
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
   * @returns ImageDTO The image was uploaded successfully
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
  }): CancelablePromise<ImageDTO> {
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
   * Get Image Full
   * Gets a full-resolution image file
   * @returns any Return the full-resolution image
   * @throws ApiError
   */
  public static getImageFull({
    imageType,
    imageName,
  }: {
    /**
     * The type of full-resolution image file to get
     */
    imageType: ImageType,
    /**
     * The name of full-resolution image file to get
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

  /**
   * Get Image Metadata
   * Gets an image's metadata
   * @returns ImageDTO Successful Response
   * @throws ApiError
   */
  public static getImageMetadata({
    imageType,
    imageName,
  }: {
    /**
     * The type of image to get
     */
    imageType: ImageType,
    /**
     * The name of image to get
     */
    imageName: string,
  }): CancelablePromise<ImageDTO> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_type}/{image_name}/metadata',
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
   * Get Image Thumbnail
   * Gets a thumbnail image file
   * @returns any Return the image thumbnail
   * @throws ApiError
   */
  public static getImageThumbnail({
    imageType,
    imageName,
  }: {
    /**
     * The type of thumbnail image file to get
     */
    imageType: ImageType,
    /**
     * The name of thumbnail image file to get
     */
    imageName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_type}/{image_name}/thumbnail',
      path: {
        'image_type': imageType,
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
    imageType,
    imageName,
  }: {
    /**
     * The type of the image whose URL to get
     */
    imageType: ImageType,
    /**
     * The name of the image whose URL to get
     */
    imageName: string,
  }): CancelablePromise<ImageUrlsDTO> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/images/{image_type}/{image_name}/urls',
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
