/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
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

}
