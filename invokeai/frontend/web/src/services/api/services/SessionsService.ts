/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { BlurInvocation } from '../models/BlurInvocation';
import type { CollectInvocation } from '../models/CollectInvocation';
import type { CropImageInvocation } from '../models/CropImageInvocation';
import type { CvInpaintInvocation } from '../models/CvInpaintInvocation';
import type { Graph } from '../models/Graph';
import type { GraphExecutionState } from '../models/GraphExecutionState';
import type { GraphInvocation } from '../models/GraphInvocation';
import type { ImageToImageInvocation } from '../models/ImageToImageInvocation';
import type { InpaintInvocation } from '../models/InpaintInvocation';
import type { InverseLerpInvocation } from '../models/InverseLerpInvocation';
import type { IterateInvocation } from '../models/IterateInvocation';
import type { LerpInvocation } from '../models/LerpInvocation';
import type { LoadImageInvocation } from '../models/LoadImageInvocation';
import type { MaskFromAlphaInvocation } from '../models/MaskFromAlphaInvocation';
import type { PaginatedResults_GraphExecutionState_ } from '../models/PaginatedResults_GraphExecutionState_';
import type { PasteImageInvocation } from '../models/PasteImageInvocation';
import type { RestoreFaceInvocation } from '../models/RestoreFaceInvocation';
import type { ShowImageInvocation } from '../models/ShowImageInvocation';
import type { TextToImageInvocation } from '../models/TextToImageInvocation';
import type { UpscaleInvocation } from '../models/UpscaleInvocation';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class SessionsService {

  /**
   * List Sessions
   * Gets a list of sessions, optionally searching
   * @param page The page of results to get
   * @param perPage The number of results per page
   * @param query The query string to search for
   * @returns PaginatedResults_GraphExecutionState_ Successful Response
   * @throws ApiError
   */
  public static listSessions(
    page?: number,
    perPage: number = 10,
    query: string = '',
  ): CancelablePromise<PaginatedResults_GraphExecutionState_> {
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
   * @param requestBody
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static createSession(
    requestBody?: Graph,
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session to get
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static getSession(
    sessionId: string,
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session
   * @param requestBody
   * @returns string Successful Response
   * @throws ApiError
   */
  public static addNode(
    sessionId: string,
    requestBody: (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | CvInpaintInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | ImageToImageInvocation | InpaintInvocation),
  ): CancelablePromise<string> {
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
   * @param sessionId The id of the session
   * @param nodePath The path to the node in the graph
   * @param requestBody
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static updateNode(
    sessionId: string,
    nodePath: string,
    requestBody: (LoadImageInvocation | ShowImageInvocation | CropImageInvocation | PasteImageInvocation | MaskFromAlphaInvocation | BlurInvocation | LerpInvocation | InverseLerpInvocation | CvInpaintInvocation | UpscaleInvocation | RestoreFaceInvocation | TextToImageInvocation | GraphInvocation | IterateInvocation | CollectInvocation | ImageToImageInvocation | InpaintInvocation),
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session
   * @param nodePath The path to the node to delete
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static deleteNode(
    sessionId: string,
    nodePath: string,
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session
   * @param requestBody
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static addEdge(
    sessionId: string,
    requestBody: Array<any>,
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session
   * @param fromNodeId The id of the node the edge is coming from
   * @param fromField The field of the node the edge is coming from
   * @param toNodeId The id of the node the edge is going to
   * @param toField The field of the node the edge is going to
   * @returns GraphExecutionState Successful Response
   * @throws ApiError
   */
  public static deleteEdge(
    sessionId: string,
    fromNodeId: string,
    fromField: string,
    toNodeId: string,
    toField: string,
  ): CancelablePromise<GraphExecutionState> {
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
   * @param sessionId The id of the session to invoke
   * @param all Whether or not to invoke all remaining invocations
   * @returns any Successful Response
   * @throws ApiError
   */
  public static invokeSession(
    sessionId: string,
    all: boolean = false,
  ): CancelablePromise<any> {
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

}
