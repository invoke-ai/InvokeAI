/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_append_invocation } from '../models/Body_append_invocation';
import type { InvocationGraph } from '../models/InvocationGraph';
import type { InvocationSession } from '../models/InvocationSession';
import type { PaginatedSession } from '../models/PaginatedSession';

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class SessionsService {

    /**
     * List Sessions
     * Gets a paged list of sessions ids
     * @param page The page of results to get
     * @param perPage The number of results per page
     * @returns PaginatedSession Successful Response
     * @throws ApiError
     */
    public static listSessions(
        page?: number,
        perPage: number = 10,
    ): CancelablePromise<PaginatedSession> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/sessions/',
            query: {
                'page': page,
                'per_page': perPage,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }

    /**
     * Create Session
     * Creates a new sessions, optionally initializing it with an invocation graph
     * @param requestBody
     * @returns InvocationSession Successful Response
     * @throws ApiError
     */
    public static createSession(
        requestBody?: InvocationGraph,
    ): CancelablePromise<InvocationSession> {
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
     * Gets a single session
     * @param sessionId The id of the session to get
     * @returns InvocationSession Successful Response
     * @throws ApiError
     */
    public static getSession(
        sessionId: string,
    ): CancelablePromise<InvocationSession> {
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
     * Append Invocation
     * @param sessionId The id of the sessions to invoke
     * @param requestBody
     * @returns InvocationSession Successful Response
     * @throws ApiError
     */
    public static appendInvocation(
        sessionId: string,
        requestBody: Body_append_invocation,
    ): CancelablePromise<InvocationSession> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/v1/sessions/{session_id}/invocations',
            path: {
                'session_id': sessionId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                400: `Invalid invocation or link`,
                404: `Session not found`,
                422: `Validation Error`,
            },
        });
    }

    /**
     * Invoke Session
     * Invokes the session
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
