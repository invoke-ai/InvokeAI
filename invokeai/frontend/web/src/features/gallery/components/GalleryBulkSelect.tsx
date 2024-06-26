import { Flex, IconButton, Spacer, Tag, TagCloseButton, TagLabel, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BiSelectMultiple } from 'react-icons/bi';

import { GallerySearch } from './GallerySearch';

export const GalleryBulkSelect = () => {
  const dispatch = useAppDispatch();
  const { selection } = useAppSelector((s) => s.gallery);
  const { t } = useTranslation();
  const { imageDTOs } = useGalleryImages();

  const onClickClearSelection = useCallback(() => {
    dispatch(selectionChanged([]));
  }, [dispatch]);

  const onClickSelectAllPage = useCallback(() => {
    dispatch(selectionChanged(selection.concat(imageDTOs)));
  }, [dispatch, imageDTOs, selection]);

  return (
    <Flex alignItems="center" justifyContent="space-between">
      <Flex>
        {selection.length > 0 ? (
          <Tag>
            <TagLabel>
              {selection.length} {t('common.selected')}
            </TagLabel>
            <Tooltip label="Clear selection">
              <TagCloseButton onClick={onClickClearSelection} />
            </Tooltip>
          </Tag>
        ) : (
          <Spacer />
        )}

        <Tooltip label={t('gallery.selectAllOnPage')}>
          <IconButton
            variant="outline"
            size="sm"
            icon={<BiSelectMultiple />}
            aria-label="Bulk select"
            onClick={onClickSelectAllPage}
          />
        </Tooltip>
      </Flex>

      <GallerySearch />
    </Flex>
  );
};
