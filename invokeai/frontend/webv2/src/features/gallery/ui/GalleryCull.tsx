import type { GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { HStack, Icon } from '@chakra-ui/react';
import { getGalleryCullMode } from '@features/gallery/core/cull';
import { galleryItemNamesOptions, galleryItemsTotalOptions } from '@features/gallery/data/queries';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Trash2Icon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

/** The listing's board, view, media kind, date range and search, listing every rejected item whatever marks filter. */
const getRejectedFilter = (filter: GalleryItemsFilter): GalleryItemsFilter => ({
  ...filter,
  colorLabels: undefined,
  flags: ['reject'],
  maxRating: undefined,
  minRating: undefined,
  starred: undefined,
});

/**
 * Cull mode's "Delete N rejected…": every rejected item of that scope, loaded or not, through the deletion flow, which
 * checks usage and always confirms a request made this way.
 */
export const GalleryCullRow = () => {
  const { t } = useTranslation();
  const { filter, gallery, itemActions } = useGalleryWidget();
  const { galleryValues, notifications } = useGalleryUi();
  const queryClient = useQueryClient();
  const [isCollecting, setIsCollecting] = useState(false);
  // A ranked similarity result keeps no marks.
  const isApplicable = getGalleryCullMode(galleryValues).enabled && gallery.semanticImageQuery === null;
  const rejectedFilter = useMemo(() => getRejectedFilter(filter), [filter]);
  // A total query sits under the listings' key, so every mark or deletion's invalidation refetches it.
  const { data: count = 0 } = useQuery({ ...galleryItemsTotalOptions(rejectedFilter), enabled: isApplicable });
  const deleteRejected = useCallback(
    async (event: { currentTarget: HTMLButtonElement }) => {
      const owner = captureAccountScope();
      const trigger = event.currentTarget;
      let refs: GalleryItemRef[];

      setIsCollecting(true);
      try {
        // Fresh: a mark made a moment ago has not reached a cached list yet.
        refs = [...(await queryClient.fetchQuery({ ...galleryItemNamesOptions(rejectedFilter), staleTime: 0 })).items];
      } catch {
        if (isAccountScopeCurrent(owner)) {
          notifications.reportError({
            area: 'cull-delete-rejected',
            message: t('widgets.gallery.cull.deleteRejectedFailed'),
            namespace: 'gallery',
          });
        }
        setIsCollecting(false);
        return;
      }

      try {
        if (isAccountScopeCurrent(owner) && refs.length > 0) {
          await itemActions.deleteItems(refs, {
            returnFocus: () => (trigger.isConnected ? trigger : null),
            targetsOffScreen: true,
          });
        }
      } finally {
        setIsCollecting(false);
      }
    },
    [itemActions, notifications, queryClient, rejectedFilter, t]
  );
  const handleClick = useCallback(
    (event: { currentTarget: HTMLButtonElement }) => void deleteRejected(event),
    [deleteRejected]
  );

  if (!isApplicable || count === 0) {
    return null;
  }

  return (
    <HStack data-gallery-cull-row="" minW="0">
      <Button
        color="fg.error"
        data-gallery-delete-rejected=""
        disabled={isCollecting}
        size="xs"
        variant="outline"
        onClick={handleClick}
      >
        <Icon as={Trash2Icon} boxSize="3" />
        {t('widgets.gallery.cull.deleteRejected', { count })}
      </Button>
    </HStack>
  );
};
