import { Button, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { memo, useCallback, useMemo } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { JumpToPaged } from './JumpToPaged';

// Logic adapted from v5.10.0 pagination utilities.
const range = (start: number, end: number) => {
  const length = end - start + 1;
  return Array.from({ length }, (_, idx) => idx + start);
};

const ELLIPSIS = 'ellipsis' as const;

const getRange = (currentPage: number, totalPages: number, siblingCount: number) => {
  const totalPageNumbers = Math.min(2 * siblingCount + 5, totalPages);

  const firstPageIndex = 1;
  const lastPageIndex = totalPages;

  const leftSiblingIndex = Math.max(currentPage - siblingCount, firstPageIndex);
  const rightSiblingIndex = Math.min(currentPage + siblingCount, lastPageIndex);

  const showLeftEllipsis = leftSiblingIndex > firstPageIndex + 1;
  const showRightEllipsis = rightSiblingIndex < lastPageIndex - 1;

  const itemCount = totalPageNumbers - 2;

  if (!showLeftEllipsis && showRightEllipsis) {
    const leftRange = range(1, itemCount);
    return [...leftRange, ELLIPSIS, lastPageIndex];
  }

  if (showLeftEllipsis && !showRightEllipsis) {
    const rightRange = range(lastPageIndex - itemCount + 1, lastPageIndex);
    return [firstPageIndex, ELLIPSIS, ...rightRange];
  }

  if (showLeftEllipsis && showRightEllipsis) {
    const middleRange = range(leftSiblingIndex, rightSiblingIndex);
    return [firstPageIndex, ELLIPSIS, ...middleRange, ELLIPSIS, lastPageIndex];
  }

  return range(firstPageIndex, lastPageIndex) as (number | 'ellipsis')[];
};

type GalleryPaginationPagedProps = {
  pageIndex: number;
  pageCount: number;
  onPrev: () => void;
  onNext: () => void;
  onGoToPage: (pageIndex: number) => void;
  onPageInputChange: (valueAsString: string, valueAsNumber: number) => void;
};

export const GalleryPaginationPaged = memo(
  ({ pageIndex, pageCount, onPrev, onNext, onGoToPage, onPageInputChange }: GalleryPaginationPagedProps) => {
    const pageButtons = useMemo(() => {
      if (pageCount > 7) {
        return getRange(pageIndex + 1, pageCount, 1);
      }
      return range(1, pageCount);
    }, [pageCount, pageIndex]);

    const onClickPrev = useCallback(() => {
      onPrev();
    }, [onPrev]);

    const onClickNext = useCallback(() => {
      onNext();
    }, [onNext]);

    if (!pageCount) {
      return null;
    }

    return (
      <Flex justifyContent="center" alignItems="center" w="full" gap={1} pt={2}>
        <IconButton
          size="sm"
          aria-label="prev"
          icon={<PiCaretLeftBold />}
          onClick={onClickPrev}
          isDisabled={pageIndex === 0}
          variant="ghost"
        />
        <Spacer />
        {pageButtons.map((page, i) => (
          <PageButton key={`${page}_${i}`} page={page} currentPage={pageIndex} goToPage={onGoToPage} />
        ))}
        <Spacer />
        <IconButton
          size="sm"
          aria-label="next"
          icon={<PiCaretRightBold />}
          onClick={onClickNext}
          isDisabled={pageIndex >= pageCount - 1}
          variant="ghost"
        />
        <JumpToPaged pageIndex={pageIndex} pageCount={pageCount} onChange={onPageInputChange} />
      </Flex>
    );
  }
);

GalleryPaginationPaged.displayName = 'GalleryPaginationPaged';

type PageButtonProps = {
  page: number | typeof ELLIPSIS;
  currentPage: number;
  goToPage: (pageIndex: number) => void;
};

const PageButton = memo(({ page, currentPage, goToPage }: PageButtonProps) => {
  const onClick = useCallback(() => {
    if (page === ELLIPSIS) {
      return;
    }
    goToPage(page - 1);
  }, [goToPage, page]);

  if (page === ELLIPSIS) {
    return (
      <Button size="sm" variant="link" isDisabled>
        ...
      </Button>
    );
  }

  return (
    <Button size="sm" onClick={onClick} variant={currentPage === page - 1 ? 'solid' : 'outline'}>
      {page}
    </Button>
  );
});

PageButton.displayName = 'PageButton';
