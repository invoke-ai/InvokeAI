/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * The origin of a resource (eg image).
 *
 * - INTERNAL: The resource was created by the application.
 * - EXTERNAL: The resource was not created by the application.
 * This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
 */
export type ResourceOrigin = 'internal' | 'external';
