/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CreateModelRequest } from '../models/CreateModelRequest';
import type { ModelsList } from '../models/ModelsList';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ModelsService {

  /**
   * List Models
   * Gets a list of models
   * @returns ModelsList Successful Response
   * @throws ApiError
   */
  public static listModels(): CancelablePromise<ModelsList> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/models/',
    });
  }

  /**
   * Update Model
   * Add Model
   * @returns any Successful Response
   * @throws ApiError
   */
  public static updateModel({
    requestBody,
  }: {
    requestBody: CreateModelRequest,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/models/',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Model
   * Delete Model
   * @returns any Successful Response
   * @throws ApiError
   */
  public static delModel({
    modelName,
  }: {
    modelName: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/models/{model_name}',
      path: {
        'model_name': modelName,
      },
      errors: {
        404: `Model not found`,
        422: `Validation Error`,
      },
    });
  }

}
