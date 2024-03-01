import { atom } from 'nanostores';

const DEFAULT_BULK_DOWNLOAD_ID = 'default';

/**
 * The download id for a bulk download. Used for socket subscriptions.
 */

export const $bulkDownloadId = atom<string>(DEFAULT_BULK_DOWNLOAD_ID);
