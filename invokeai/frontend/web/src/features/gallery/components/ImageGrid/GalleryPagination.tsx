import { Button, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { PiCaretDoubleLeftBold, PiCaretDoubleRightBold, PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

export const GalleryPagination = () => {
  const {
    goPrev,
    goNext,
    goToFirst,
    goToLast,
    isFirstEnabled,
    isLastEnabled,
    isPrevEnabled,
    isNextEnabled,
    pageButtons,
    goToPage,
    currentPage,
    rangeDisplay,
    total,
  } = useGalleryPagination();

  if (!total) {
    return <Flex flexDir="column" alignItems="center" gap="2" height="48px"></Flex>;
  }

  return (
    <Flex flexDir="column" alignItems="center" gap="2" height="48px">
      <Flex gap={2} alignItems="center" w="full">
        <IconButton
          size="sm"
          aria-label="prev"
          icon={<PiCaretDoubleLeftBold />}
          onClick={goToFirst}
          isDisabled={!isFirstEnabled}
        />
        <IconButton
          size="sm"
          aria-label="prev"
          icon={<PiCaretLeftBold />}
          onClick={goPrev}
          isDisabled={!isPrevEnabled}
        />
        <Spacer />
        {pageButtons.map((page) => (
          <Button
            size="sm"
            key={page}
            onClick={goToPage.bind(null, page)}
            variant={currentPage === page ? 'solid' : 'outline'}
          >
            {page + 1}
          </Button>
        ))}
        <Spacer />
        <IconButton
          size="sm"
          aria-label="next"
          icon={<PiCaretRightBold />}
          onClick={goNext}
          isDisabled={!isNextEnabled}
        />
        <IconButton
          size="sm"
          aria-label="next"
          icon={<PiCaretDoubleRightBold />}
          onClick={goToLast}
          isDisabled={!isLastEnabled}
        />
      </Flex>
      <Text>{rangeDisplay}</Text>
    </Flex>
  );
};
