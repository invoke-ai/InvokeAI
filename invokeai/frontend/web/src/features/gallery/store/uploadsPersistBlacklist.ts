import { UploadsState } from './uploadsSlice';

/**
 * Uploads slice persist blacklist
 *
 * Currently blacklisting uploads slice entirely, see persist config in store.ts
 */
const itemsToBlacklist: (keyof UploadsState)[] = [];

export const uploadsBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `uploads.${blacklistItem}`
);
