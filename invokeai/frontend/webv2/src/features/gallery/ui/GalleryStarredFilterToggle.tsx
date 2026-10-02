import { ToggleIconButton } from '@platform/ui/Button';
import { StarIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryWidget } from './GalleryWidgetContext';

/**
 * Restricts the listing to starred items. Inert (but still focusable, so the
 * tooltip can explain why) under a ranked similarity result, which the
 * starred filter does not apply to.
 */
export const GalleryStarredFilterToggle = () => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const isRanked = gallery.semanticImageQuery !== null;
  const handleCheckedChange = useCallback(
    (checked: boolean) => {
      if (!isRanked) {
        actions.setStarredOnly(checked);
      }
    },
    [actions, isRanked]
  );

  return (
    <ToggleIconButton
      aria-disabled={isRanked || undefined}
      checked={gallery.starredOnly && !isRanked}
      data-gallery-starred-filter-toggle
      flexShrink={0}
      icon={StarIcon}
      label={t('widgets.gallery.starredOnly')}
      size="xs"
      tooltip={t(isRanked ? 'widgets.gallery.starredOnlyUnavailable' : 'widgets.gallery.starredOnlyTooltip')}
      onCheckedChange={handleCheckedChange}
    />
  );
};
