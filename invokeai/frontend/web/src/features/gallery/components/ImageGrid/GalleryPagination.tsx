import { Button, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useGalleryPagination } from '../../hooks/useGalleryPagination';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

export const GalleryPagination = () => {
  const { goPrev, goNext, isPrevEnabled, isNextEnabled, pageButtons, goToPage, currentPage, rangeDisplay } =
    useGalleryPagination();
  console.log({ currentPage, pageButtons });

  return (
    <Flex flexDir="column" alignItems="center" gap="2">
      <Flex gap={2} alignItems="flex-end">
        <IconButton
          size="sm"
          aria-label="prev"
          icon={<PiCaretLeftBold />}
          onClick={goPrev}
          isDisabled={!isPrevEnabled}
        />
        {pageButtons.map((page) =>
          typeof page === 'number' ? (
            <Button
              size="sm"
              key={page}
              onClick={goToPage.bind(null, page)}
              variant={currentPage === page ? 'solid' : 'outline'}
            >
              {page + 1}
            </Button>
          ) : (
            <Text fontSize="md">...</Text>
          )
        )}
        <IconButton
          size="sm"
          aria-label="next"
          icon={<PiCaretRightBold />}
          onClick={goNext}
          isDisabled={!isNextEnabled}
        />
      </Flex>
      <Text>{rangeDisplay} Images</Text>
    </Flex>
  );
};
