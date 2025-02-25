import { Button, Flex, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { ELLIPSIS, useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { useSelectAll } from 'features/gallery/hooks/useSelectAll';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { BiSelectMultiple } from 'react-icons/bi';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { JumpTo } from './JumpTo';

export const GalleryPagination = memo(() => {
  const { t } = useTranslation();
  const selectAll = useSelectAll();
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
      <Tooltip label={`Select all (${total})`}>
        <IconButton
          variant="outline"
          size="sm"
          icon={<BiSelectMultiple />}
          aria-label={t('gallery.selectAll')}
          onClick={selectAll}
        />
      </Tooltip>
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
