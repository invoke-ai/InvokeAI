import { UploadsState } from './uploadsSlice';

/**
 * Uploads slice persist denylist
 *
 * Currently denylisting uploads slice entirely, see persist config in store.ts
 */
const itemsToDenylist: (keyof UploadsState)[] = ['isLoading'];

export const uploadsDenylist = itemsToDenylist.map(
  (denylistItem) => `uploads.${denylistItem}`
);
