import { isAcceptedUploadFile } from 'common/util/uploadMediaAccept';
import { z } from 'zod';

/**
 * Validates files entering via the fullscreen drag-drop / paste path.
 *
 * MIME type and filename extension each suffice on their own (`isAcceptedUploadFile`):
 * browsers sometimes supply an empty or generic `File.type` — e.g. a `clip.mp4` dragged
 * from some file managers — and requiring both signals used to reject files the backend
 * upload routes would happily accept.
 */
export const zUploadFile = z.custom<File>().refine(isAcceptedUploadFile, { message: 'File type is not supported' });
