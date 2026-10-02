/** Media names referenced by persisted and in-memory Workbench state, matched by key like the server's extractor. */

/** Keys whose string values name an image; the server's `media_references` extractor uses the same set. */
export const IMAGE_NAME_KEYS: ReadonlySet<string> = new Set(['imageName', 'image_name']);

/** Keys whose string values name a video; the server's `media_references` extractor uses the same set. */
export const VIDEO_NAME_KEYS: ReadonlySet<string> = new Set(['videoName', 'video_name', 'source_video_name']);

/** Server-side names are generated filenames; the cleanup index ignores anything longer. */
export const MAX_HELD_NAME_LENGTH = 255;

export interface MediaNameRefs {
  images: Set<string>;
  videos: Set<string>;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Every name under a media-name key, at any depth, except subtrees whose key `skipKey` rejects. */
export const collectMediaNames = (
  roots: readonly unknown[],
  skipKey: (key: string) => boolean,
  maxLength = Number.POSITIVE_INFINITY
): MediaNameRefs => {
  const refs: MediaNameRefs = { images: new Set<string>(), videos: new Set<string>() };
  const pending = [...roots];
  while (pending.length > 0) {
    const node = pending.pop();
    if (Array.isArray(node)) {
      for (const item of node) {
        pending.push(item);
      }
      continue;
    }
    if (!isRecord(node)) {
      continue;
    }
    for (const [key, value] of Object.entries(node)) {
      if (skipKey(key)) {
        continue;
      }
      if (typeof value === 'string') {
        if (value !== '' && value.length <= maxLength) {
          if (IMAGE_NAME_KEYS.has(key)) {
            refs.images.add(value);
          } else if (VIDEO_NAME_KEYS.has(key)) {
            refs.videos.add(value);
          }
        }
      } else if (typeof value === 'object' && value !== null) {
        pending.push(value);
      }
    }
  }
  return refs;
};

/** Every media name in `values`, history included: what an undo entry can restore. */
export const collectRestorableAssetRefs = (...values: unknown[]): MediaNameRefs =>
  collectMediaNames(values, () => false, MAX_HELD_NAME_LENGTH);
