import { UploadsState } from './uploadsSlice';

/**
 * Uploads slice persist denylist
 *
 * Currently denylisting uploads slice entirely, see `serialize.ts`
 */
export const uploadsPersistDenylist: (keyof UploadsState)[] = [];
