import { Button, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { ELLIPSIS, useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { useCallback } from 'react';
import { PiCaretLeftBold, PiCaretRightBold, PiDotsThreeBold } from 'react-icons/pi';

import { JumpTo } from './JumpTo';

export const GalleryPagination = () => {
  const { goPrev, goNext, isPrevEnabled, isNextEnabled, pageButtons, goToPage, currentPage, total } =
    useGalleryPagination();

  const onClickPrev = useCallback(() => {
    goPrev();
  }, [goPrev]);

  const onClickNext = useCallback(() => {
    goNext();
  }, [goNext]);

  if (!total) {
    return null;
  }

  return (
    <Flex justifyContent="center" alignItems="center" w="full" gap={1}>
      <IconButton
        size="sm"
        aria-label="prev"
        icon={<PiCaretLeftBold />}
        onClick={onClickPrev}
        isDisabled={!isPrevEnabled}
        variant="ghost"
      />
      {pageButtons.map((page, i) => {
        if (page === ELLIPSIS) {
          return <Icon as={PiDotsThreeBold} boxSize="4" key={`ellipsis-${i}`} />;
        }
        return (
          <Button
            size="sm"
            key={page}
            onClick={goToPage.bind(null, page - 1)}
            variant={currentPage === page - 1 ? 'solid' : 'outline'}
          >
            {page}
          </Button>
        );
      })}
      <IconButton
        size="sm"
        aria-label="next"
        icon={<PiCaretRightBold />}
        onClick={onClickNext}
        isDisabled={!isNextEnabled}
        variant="ghost"
      />
      <JumpTo />
    </Flex>
  );
};
