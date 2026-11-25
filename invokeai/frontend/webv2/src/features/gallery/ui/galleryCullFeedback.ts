import type { GalleryMarksPatch } from '@features/gallery/core/marks';
import type { TFunction } from 'i18next';

import { getGalleryColorLabelName } from './GalleryItemMarks';

/** What a cull key did, as the Preview HUD shows it (`text`) and as it is announced. */
export interface GalleryCullFeedback {
  announcement: string;
  text: string;
}

const getMarkFeedback = (t: TFunction, patch: GalleryMarksPatch): GalleryCullFeedback | null => {
  if (patch.rating !== undefined) {
    return patch.rating === 0
      ? {
          announcement: t('widgets.gallery.cull.feedback.ratingCleared'),
          text: t('widgets.gallery.cull.feedback.ratingCleared'),
        }
      : {
          announcement: t('widgets.gallery.marks.ratingSummary', { count: patch.rating }),
          text: '★'.repeat(patch.rating),
        };
  }
  if (patch.colorLabel !== undefined) {
    const text = patch.colorLabel
      ? t('widgets.gallery.marks.colorLabel', { label: getGalleryColorLabelName(t, patch.colorLabel) })
      : t('widgets.gallery.cull.feedback.labelCleared');

    return { announcement: text, text };
  }
  if (patch.flag !== undefined) {
    const key = patch.flag === 'pick' ? 'bookmarked' : patch.flag === 'reject' ? 'rejected' : 'flagCleared';
    const text = t(`widgets.gallery.cull.feedback.${key}`);

    return { announcement: text, text };
  }

  return null;
};

/** A null patch had no target: the key still answers, saying there is nothing to mark. */
export const getGalleryCullFeedback = (
  t: TFunction,
  patch: GalleryMarksPatch | null,
  targetCount: number
): GalleryCullFeedback | null => {
  if (patch === null) {
    const text = t('widgets.gallery.cull.feedback.nothingToMark');

    return { announcement: text, text };
  }

  const feedback = getMarkFeedback(t, patch);

  return feedback && targetCount > 1
    ? {
        ...feedback,
        announcement: t('widgets.gallery.cull.feedback.forItems', {
          count: targetCount,
          mark: feedback.announcement,
        }),
      }
    : feedback;
};
