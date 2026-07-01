import type { WidgetViewProps } from '@workbench/types';

import { ButtonGroup, HStack, Pagination } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { getGallerySettings } from '@workbench/gallery/settings';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { ChevronLeftIcon, ChevronRightIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { getGalleryPage, getGalleryTotalImages } from './galleryStateView';
import { GALLERY_PAGE_SIZE } from './useGalleryData';

const PAGINATION_ITEM_VARIANT = { _selected: 'outline', base: 'ghost' } as const;

/**
 * Widget-chrome footer: page navigation for the gallery's paginated mode.
 * Reads everything from the gallery widget's persisted values so it stays
 * decoupled from the view's data fetching.
 */
export const GalleryWidgetFooter = (_props: WidgetViewProps) => {
  const { t } = useTranslation();
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const dispatch = useWorkbenchDispatch();
  const settings = getGallerySettings(galleryValues);
  const page = getGalleryPage(galleryValues);
  const totalImages = getGalleryTotalImages(galleryValues);
  const handlePageChange = useCallback(
    (event: { page: number }) => dispatch({ page: event.page - 1, type: 'setGalleryPage' }),
    [dispatch]
  );
  const renderPaginationItem = useCallback(
    (paginationPage: { value: number }) => (
      <IconButton aria-label={t('common.pageNumber', { page: paginationPage.value })} variant={PAGINATION_ITEM_VARIANT}>
        {paginationPage.value}
      </IconButton>
    ),
    [t]
  );

  if (settings.paginationMode !== 'paginated' || totalImages === null || totalImages <= GALLERY_PAGE_SIZE) {
    return null;
  }

  return (
    <HStack justify="center" py="1" w="full">
      <Pagination.Root
        count={totalImages}
        page={page + 1}
        pageSize={GALLERY_PAGE_SIZE}
        siblingCount={1}
        onPageChange={handlePageChange}
      >
        <ButtonGroup gap="1" size="2xs" variant="ghost">
          <Pagination.PrevTrigger asChild>
            <IconButton aria-label={t('common.previousPage')}>
              <ChevronLeftIcon />
            </IconButton>
          </Pagination.PrevTrigger>
          <Pagination.Items render={renderPaginationItem} />
          <Pagination.NextTrigger asChild>
            <IconButton aria-label={t('common.nextPage')}>
              <ChevronRightIcon />
            </IconButton>
          </Pagination.NextTrigger>
        </ButtonGroup>
      </Pagination.Root>
    </HStack>
  );
};
