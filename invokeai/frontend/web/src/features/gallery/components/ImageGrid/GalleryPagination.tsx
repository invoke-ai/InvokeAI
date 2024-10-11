import { Button, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { ELLIPSIS, useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { memo, useCallback } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { JumpTo } from './JumpTo';

export const GalleryPagination = memo(() => {
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
    <Flex justifyContent="center" alignItems="center" w="full" gap={1} pt={2}>
      <IconButton
        size="sm"
        aria-label="prev"
        icon={<PiCaretLeftBold />}
        onClick={onClickPrev}
        isDisabled={!isPrevEnabled}
        variant="ghost"
      />
      <Spacer />
      {pageButtons.map((page, i) => (
        <PageButton key={`${page}_${i}`} page={page} currentPage={currentPage} goToPage={goToPage} />
      ))}
      <Spacer />
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
});

GalleryPagination.displayName = 'GalleryPagination';

type PageButtonProps = {
  page: number | typeof ELLIPSIS;
  currentPage: number;
  goToPage: (page: number) => void;
};

const PageButton = memo(({ page, currentPage, goToPage }: PageButtonProps) => {
  if (page === ELLIPSIS) {
    return (
      <Button size="sm" variant="link" isDisabled>
        ...
      </Button>
    );
  }
  return (
    <Button size="sm" onClick={goToPage.bind(null, page - 1)} variant={currentPage === page - 1 ? 'solid' : 'outline'}>
      {page}
    </Button>
  );
});

PageButton.displayName = 'PageButton';
