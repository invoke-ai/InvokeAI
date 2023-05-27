/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * The category of an image.
 *
 * - GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose.
 * - MASK: The image is a mask image.
 * - CONTROL: The image is a ControlNet control image.
 * - USER: The image is a user-provide image.
 * - OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes.
 */
export type ImageCategory = 'general' | 'mask' | 'control' | 'user' | 'other';
