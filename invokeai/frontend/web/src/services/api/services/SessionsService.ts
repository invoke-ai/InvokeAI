/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AddInvocation } from '../models/AddInvocation';
import type { CannyImageProcessorInvocation } from '../models/CannyImageProcessorInvocation';
import type { CollectInvocation } from '../models/CollectInvocation';
import type { CompelInvocation } from '../models/CompelInvocation';
import type { ContentShuffleImageProcessorInvocation } from '../models/ContentShuffleImageProcessorInvocation';
import type { ControlNetInvocation } from '../models/ControlNetInvocation';
import type { CvInpaintInvocation } from '../models/CvInpaintInvocation';
import type { DivideInvocation } from '../models/DivideInvocation';
import type { Edge } from '../models/Edge';
import type { Graph } from '../models/Graph';
import type { GraphExecutionState } from '../models/GraphExecutionState';
import type { GraphInvocation } from '../models/GraphInvocation';
import type { HedImageprocessorInvocation } from '../models/HedImageprocessorInvocation';
import type { ImageBlurInvocation } from '../models/ImageBlurInvocation';
import type { ImageChannelInvocation } from '../models/ImageChannelInvocation';
import type { ImageConvertInvocation } from '../models/ImageConvertInvocation';
import type { ImageCropInvocation } from '../models/ImageCropInvocation';
import type { ImageInverseLerpInvocation } from '../models/ImageInverseLerpInvocation';
import type { ImageLerpInvocation } from '../models/ImageLerpInvocation';
import type { ImageMultiplyInvocation } from '../models/ImageMultiplyInvocation';
import type { ImagePasteInvocation } from '../models/ImagePasteInvocation';
import type { ImageProcessorInvocation } from '../models/ImageProcessorInvocation';
import type { ImageResizeInvocation } from '../models/ImageResizeInvocation';
import type { ImageScaleInvocation } from '../models/ImageScaleInvocation';
import type { ImageToImageInvocation } from '../models/ImageToImageInvocation';
import type { ImageToLatentsInvocation } from '../models/ImageToLatentsInvocation';
import type { InfillColorInvocation } from '../models/InfillColorInvocation';
import type { InfillPatchMatchInvocation } from '../models/InfillPatchMatchInvocation';
import type { InfillTileInvocation } from '../models/InfillTileInvocation';
import type { InpaintInvocation } from '../models/InpaintInvocation';
import type { IterateInvocation } from '../models/IterateInvocation';
import type { LatentsToImageInvocation } from '../models/LatentsToImageInvocation';
import type { LatentsToLatentsInvocation } from '../models/LatentsToLatentsInvocation';
import type { LineartAnimeImageProcessorInvocation } from '../models/LineartAnimeImageProcessorInvocation';
import type { LineartImageProcessorInvocation } from '../models/LineartImageProcessorInvocation';
import type { LoadImageInvocation } from '../models/LoadImageInvocation';
import type { MaskFromAlphaInvocation } from '../models/MaskFromAlphaInvocation';
import type { MediapipeFaceProcessorInvocation } from '../models/MediapipeFaceProcessorInvocation';
import type { MidasDepthImageProcessorInvocation } from '../models/MidasDepthImageProcessorInvocation';
import type { MlsdImageProcessorInvocation } from '../models/MlsdImageProcessorInvocation';
import type { MultiplyInvocation } from '../models/MultiplyInvocation';
import type { NoiseInvocation } from '../models/NoiseInvocation';
import type { NormalbaeImageProcessorInvocation } from '../models/NormalbaeImageProcessorInvocation';
import type { OpenposeImageProcessorInvocation } from '../models/OpenposeImageProcessorInvocation';
import type { PaginatedResults_GraphExecutionState_ } from '../models/PaginatedResults_GraphExecutionState_';
import type { ParamFloatInvocation } from '../models/ParamFloatInvocation';
import type { ParamIntInvocation } from '../models/ParamIntInvocation';
import type { PidiImageProcessorInvocation } from '../models/PidiImageProcessorInvocation';
import type { RandomIntInvocation } from '../models/RandomIntInvocation';
import type { RandomRangeInvocation } from '../models/RandomRangeInvocation';
import type { RangeInvocation } from '../models/RangeInvocation';
import type { RangeOfSizeInvocation } from '../models/RangeOfSizeInvocation';
import type { ResizeLatentsInvocation } from '../models/ResizeLatentsInvocation';
import type { RestoreFaceInvocation } from '../models/RestoreFaceInvocation';
import type { ScaleLatentsInvocation } from '../models/ScaleLatentsInvocation';
import type { ShowImageInvocation } from '../models/ShowImageInvocation';
import type { SubtractInvocation } from '../models/SubtractInvocation';
import type { TextToImageInvocation } from '../models/TextToImageInvocation';
import type { TextToLatentsInvocation } from '../models/TextToLatentsInvocation';
import type { UpscaleInvocation } from '../models/UpscaleInvocation';
import type { ZoeDepthImageProcessorInvocation } from '../models/ZoeDepthImageProcessorInvocation';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class SessionsService {

  /**
   * List Sessions
   * Gets a list of sessions, optionally searching
   * @returns PaginatedResults_GraphExecutionState_ Successful Response
   * @throws ApiError
   */
  public static listSessions({
    page,
    perPage = 10,
    query = '',
  }: {
    /**
     * The page of results to get
     */
    page?: number,
    /**
     * The number of results per page
     */
    perPage?: number,
    /**
     * The query string to search for
     */
    query?: string,
  }): CancelablePromise<PaginatedResults_GraphExecutionState_> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/sessions/',
      query: {
        'page': page,
        'per_page': perPage,
        'query': query,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

  /**
   * Create Session
   * Creates a new session, optionally initializing it with an invocation graph
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static createSession({
    requestBody,
  }: {
    requestBody?: Graph,
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/sessions/',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        400: `Invalid json`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Get Session
   * Gets a session
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static getSession({
    sessionId,
  }: {
    /**
     * The id of the session to get
     */
    sessionId: string,
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/api/v1/sessions/{session_id}',
      path: {
        'session_id': sessionId,
      },
      errors: {
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Add Node
   * Adds a node to the graph
   * @returns string Successful Response
   * @throws ApiError
   */
  public static addNode({
    sessionId,
    requestBody,
  }: {
    /**
     * The id of the session
     */
    sessionId: string,
    requestBody: (LoadImageInvocation | ShowImageInvocation | ImageCropInvocation | ImagePasteInvocation | MaskFromAlphaInvocation | ImageMultiplyInvocation | ImageChannelInvocation | ImageConvertInvocation | ImageBlurInvocation | ImageResizeInvocation | ImageScaleInvocation | ImageLerpInvocation | ImageInverseLerpInvocation | ControlNetInvocation | ImageProcessorInvocation | CompelInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | RandomIntInvocation | ParamIntInvocation | ParamFloatInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | ImageToLatentsInvocation | CvInpaintInvocation | RangeInvocation | RangeOfSizeInvocation | RandomRangeInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | InfillColorInvocation | InfillTileInvocation | InfillPatchMatchInvocation | GraphInvocation | IterateInvocation | CollectInvocation | CannyImageProcessorInvocation | HedImageprocessorInvocation | LineartImageProcessorInvocation | LineartAnimeImageProcessorInvocation | OpenposeImageProcessorInvocation | MidasDepthImageProcessorInvocation | NormalbaeImageProcessorInvocation | MlsdImageProcessorInvocation | PidiImageProcessorInvocation | ContentShuffleImageProcessorInvocation | ZoeDepthImageProcessorInvocation | MediapipeFaceProcessorInvocation | LatentsToLatentsInvocation | ImageToImageInvocation | InpaintInvocation),
  }): CancelablePromise<string> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/sessions/{session_id}/nodes',
      path: {
        'session_id': sessionId,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        400: `Invalid node or link`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Update Node
   * Updates a node in the graph and removes all linked edges
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static updateNode({
    sessionId,
    nodePath,
    requestBody,
  }: {
    /**
     * The id of the session
     */
    sessionId: string,
    /**
     * The path to the node in the graph
     */
    nodePath: string,
    requestBody: (LoadImageInvocation | ShowImageInvocation | ImageCropInvocation | ImagePasteInvocation | MaskFromAlphaInvocation | ImageMultiplyInvocation | ImageChannelInvocation | ImageConvertInvocation | ImageBlurInvocation | ImageResizeInvocation | ImageScaleInvocation | ImageLerpInvocation | ImageInverseLerpInvocation | ControlNetInvocation | ImageProcessorInvocation | CompelInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | RandomIntInvocation | ParamIntInvocation | ParamFloatInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | ImageToLatentsInvocation | CvInpaintInvocation | RangeInvocation | RangeOfSizeInvocation | RandomRangeInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | InfillColorInvocation | InfillTileInvocation | InfillPatchMatchInvocation | GraphInvocation | IterateInvocation | CollectInvocation | CannyImageProcessorInvocation | HedImageprocessorInvocation | LineartImageProcessorInvocation | LineartAnimeImageProcessorInvocation | OpenposeImageProcessorInvocation | MidasDepthImageProcessorInvocation | NormalbaeImageProcessorInvocation | MlsdImageProcessorInvocation | PidiImageProcessorInvocation | ContentShuffleImageProcessorInvocation | ZoeDepthImageProcessorInvocation | MediapipeFaceProcessorInvocation | LatentsToLatentsInvocation | ImageToImageInvocation | InpaintInvocation),
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'PUT',
      url: '/api/v1/sessions/{session_id}/nodes/{node_path}',
      path: {
        'session_id': sessionId,
        'node_path': nodePath,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        400: `Invalid node or link`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Node
   * Deletes a node in the graph and removes all linked edges
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static deleteNode({
    sessionId,
    nodePath,
  }: {
    /**
     * The id of the session
     */
    sessionId: string,
    /**
     * The path to the node to delete
     */
    nodePath: string,
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/sessions/{session_id}/nodes/{node_path}',
      path: {
        'session_id': sessionId,
        'node_path': nodePath,
      },
      errors: {
        400: `Invalid node or link`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Add Edge
   * Adds an edge to the graph
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static addEdge({
    sessionId,
    requestBody,
  }: {
    /**
     * The id of the session
     */
    sessionId: string,
    requestBody: Edge,
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/api/v1/sessions/{session_id}/edges',
      path: {
        'session_id': sessionId,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        400: `Invalid node or link`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Delete Edge
   * Deletes an edge from the graph
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static deleteEdge({
    sessionId,
    fromNodeId,
    fromField,
    toNodeId,
    toField,
  }: {
    /**
     * The id of the session
     */
    sessionId: string,
    /**
     * The id of the node the edge is coming from
     */
    fromNodeId: string,
    /**
     * The field of the node the edge is coming from
     */
    fromField: string,
    /**
     * The id of the node the edge is going to
     */
    toNodeId: string,
    /**
     * The field of the node the edge is going to
     */
    toField: string,
  }): CancelablePromise<GraphExecutionState> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/sessions/{session_id}/edges/{from_node_id}/{from_field}/{to_node_id}/{to_field}',
      path: {
        'session_id': sessionId,
        'from_node_id': fromNodeId,
        'from_field': fromField,
        'to_node_id': toNodeId,
        'to_field': toField,
      },
      errors: {
        400: `Invalid node or link`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Invoke Session
   * Invokes a session
   * @returns any Successful Response
   * @throws ApiError
   */
  public static invokeSession({
    sessionId,
    all = false,
  }: {
    /**
     * The id of the session to invoke
     */
    sessionId: string,
    /**
     * Whether or not to invoke all remaining invocations
     */
    all?: boolean,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'PUT',
      url: '/api/v1/sessions/{session_id}/invoke',
      path: {
        'session_id': sessionId,
      },
      query: {
        'all': all,
      },
      errors: {
        400: `The session has no invocations ready to invoke`,
        404: `Session not found`,
        422: `Validation Error`,
      },
    });
  }

  /**
   * Cancel Session Invoke
   * Invokes a session
   * @returns any Successful Response
   * @throws ApiError
   */
  public static cancelSessionInvoke({
    sessionId,
  }: {
    /**
     * The id of the session to cancel
     */
    sessionId: string,
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/api/v1/sessions/{session_id}/invoke',
      path: {
        'session_id': sessionId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }

}
