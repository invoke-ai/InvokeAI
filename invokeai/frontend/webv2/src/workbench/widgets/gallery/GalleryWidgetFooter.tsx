import { ButtonGroup, HStack, Pagination } from '@chakra-ui/react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { IconButton } from '../../components/ui/Button';
import { getGallerySettings } from '../../gallery/settings';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';
import { getGalleryPage, getGalleryTotalImages } from './galleryStateView';
import { GALLERY_PAGE_SIZE } from './useGalleryData';

/**
 * Widget-chrome footer: page navigation for the gallery's paginated mode.
 * Reads everything from the gallery widget's persisted values so it stays
 * decoupled from the view's data fetching.
 */
export const GalleryWidgetFooter = (_props: WidgetViewProps) => {
  const { activeProject, dispatch } = useWorkbench();
  const galleryValues = activeProject.widgetStates.gallery.values;
  const settings = getGallerySettings(galleryValues);
  const page = getGalleryPage(galleryValues);
  const totalImages = getGalleryTotalImages(galleryValues);

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
        onPageChange={(event) => dispatch({ page: event.page - 1, type: 'setGalleryPage' })}
      >
        <ButtonGroup gap="1" size="2xs" variant="ghost">
          <Pagination.PrevTrigger asChild>
            <IconButton aria-label="Previous page">
              <PiCaretLeftBold />
            </IconButton>
          </Pagination.PrevTrigger>
          <Pagination.Items
            render={(paginationPage) => (
              <IconButton aria-label={`Page ${paginationPage.value}`} variant={{ _selected: 'outline', base: 'ghost' }}>
                {paginationPage.value}
              </IconButton>
            )}
          />
          <Pagination.NextTrigger asChild>
            <IconButton aria-label="Next page">
              <PiCaretRightBold />
            </IconButton>
          </Pagination.NextTrigger>
        </ButtonGroup>
      </Pagination.Root>
    </HStack>
  );
};
