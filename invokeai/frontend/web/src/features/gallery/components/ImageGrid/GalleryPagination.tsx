import { Button, CompositeNumberInput, Flex, Icon, IconButton, Spacer } from '@invoke-ai/ui-library';
import { ELLIPSIS, useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { useCallback } from 'react';
import { PiCaretLeftBold, PiCaretRightBold, PiDotsThreeBold } from 'react-icons/pi';

export const GalleryPagination = () => {
  const { goPrev, goNext, isPrevEnabled, isNextEnabled, pageButtons, goToPage, currentPage, total } =
    useGalleryPagination();

  const onClickPrev = useCallback(() => {
    goPrev();
  }, [goPrev]);

  const onClickNext = useCallback(() => {
    goNext();
  }, [goNext]);

  const onChangeJumpTo = useCallback(
    (v: number) => {
      goToPage(v - 1);
    },
    [goToPage]
  );

  if (!total) {
    return null;
  }

  return (
    <Flex justifyContent="space-between" alignItems="center" w="full">
      <Spacer w="auto" />
      <Flex flexGrow="1" justifyContent="center">
        <Flex gap={1} alignItems="center">
          <IconButton
            size="xs"
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
                size="xs"
                key={page}
                onClick={goToPage.bind(null, page - 1)}
                variant={currentPage === page - 1 ? 'solid' : 'outline'}
              >
                {page}
              </Button>
            );
          })}
          <IconButton
            size="xs"
            aria-label="next"
            icon={<PiCaretRightBold />}
            onClick={onClickNext}
            isDisabled={!isNextEnabled}
            variant="ghost"
          />
        </Flex>
      </Flex>
      <CompositeNumberInput
        size="xs"
        maxW="60px"
        value={currentPage + 1}
        min={1}
        max={total}
        step={1}
        onChange={onChangeJumpTo}
      />
    </Flex>
  );
};
