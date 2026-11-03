/** A translator that interpolates values and pluralizes by `count`; i18next's `t` fits. */
export type GalleryCountTranslate = (key: string, values: Record<string, number | string>) => string;

export interface GalleryMediaCounts {
  images?: number;
  videos?: number;
  /** Uploaded images and videos, which the Uploads tab lists. */
  uploads?: number;
}

/**
 * The nonzero counts as one phrase ("2 images and 1 video"), so a summary never reads "0 videos"; an empty string
 * when every count is zero. The list's joining words are translations of their own, so each language places them.
 */
export const formatGalleryMediaCounts = (t: GalleryCountTranslate, counts: GalleryMediaCounts): string => {
  const [first = '', second, third] = (
    [
      ['widgets.gallery.imageCount', counts.images ?? 0],
      ['widgets.gallery.videoCount', counts.videos ?? 0],
      ['widgets.gallery.assetCount', counts.uploads ?? 0],
    ] as const
  )
    .filter(([, count]) => count > 0)
    .map(([key, count]) => t(key, { count }));

  if (third !== undefined && second !== undefined) {
    return t('widgets.gallery.mediaCountsThree', { first, second, third });
  }

  return second === undefined ? first : t('widgets.gallery.mediaCountsTwo', { first, second });
};
