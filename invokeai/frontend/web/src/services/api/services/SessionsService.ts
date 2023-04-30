/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AddInvocation } from '../models/AddInvocation';
import type { BlurInvocation } from '../models/BlurInvocation';
import type { CollectInvocation } from '../models/CollectInvocation';
import type { CropImageInvocation } from '../models/CropImageInvocation';
import type { CvInpaintInvocation } from '../models/CvInpaintInvocation';
import type { DivideInvocation } from '../models/DivideInvocation';
import type { Edge } from '../models/Edge';
import type { Graph } from '../models/Graph';
import type { GraphExecutionState } from '../models/GraphExecutionState';
import type { GraphInvocation } from '../models/GraphInvocation';
import type { ImageToImageInvocation } from '../models/ImageToImageInvocation';
import type { InpaintInvocation } from '../models/InpaintInvocation';
import type { InverseLerpInvocation } from '../models/InverseLerpInvocation';
import type { IterateInvocation } from '../models/IterateInvocation';
import type { LatentsToImageInvocation } from '../models/LatentsToImageInvocation';
import type { LatentsToLatentsInvocation } from '../models/LatentsToLatentsInvocation';
import type { LerpInvocation } from '../models/LerpInvocation';
import type { LoadImageInvocation } from '../models/LoadImageInvocation';
import type { MaskFromAlphaInvocation } from '../models/MaskFromAlphaInvocation';
import type { MultiplyInvocation } from '../models/MultiplyInvocation';
import type { NoiseInvocation } from '../models/NoiseInvocation';
import type { PaginatedResults_GraphExecutionState_ } from '../models/PaginatedResults_GraphExecutionState_';
import type { ParamIntInvocation } from '../models/ParamIntInvocation';
import type { PasteImageInvocation } from '../models/PasteImageInvocation';
import type { RandomRangeInvocation } from '../models/RandomRangeInvocation';
import type { RangeInvocation } from '../models/RangeInvocation';
import type { ResizeLatentsInvocation } from '../models/ResizeLatentsInvocation';
import type { RestoreFaceInvocation } from '../models/RestoreFaceInvocation';
import type { ScaleLatentsInvocation } from '../models/ScaleLatentsInvocation';
import type { ShowImageInvocation } from '../models/ShowImageInvocation';
import type { SubtractInvocation } from '../models/SubtractInvocation';
import type { TextToImageInvocation } from '../models/TextToImageInvocation';
import type { TextToLatentsInvocation } from '../models/TextToLatentsInvocation';
import type { UpscaleInvocation } from '../models/UpscaleInvocation';

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
    requestBody: (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | ParamIntInvocation | CvInpaintInvocation | RangeInvocation | RandomRangeInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | LatentsToLatentsInvocation | ImageToImageInvocation | InpaintInvocation),
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
    requestBody: (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | NoiseInvocation | TextToLatentsInvocation | LatentsToImageInvocation | ResizeLatentsInvocation | ScaleLatentsInvocation | AddInvocation | SubtractInvocation | MultiplyInvocation | DivideInvocation | ParamIntInvocation | CvInpaintInvocation | RangeInvocation | RandomRangeInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | LatentsToLatentsInvocation | ImageToImageInvocation | InpaintInvocation),
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
