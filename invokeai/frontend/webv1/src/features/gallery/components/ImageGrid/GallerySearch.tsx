import { IconButton, Input, InputGroup, InputRightElement, Spinner } from '@invoke-ai/ui-library';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

type Props = {
  searchTerm: string;
  onChangeSearchTerm: (value: string) => void;
  onResetSearchTerm: () => void;
};

export const GallerySearch = memo(({ searchTerm, onChangeSearchTerm, onResetSearchTerm }: Props) => {
  const { t } = useTranslation();
  const { isFetching } = useGalleryImageNames();

  const handleChangeInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChangeSearchTerm(e.target.value);
    },
    [onChangeSearchTerm]
  );

  const handleKeydown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        onResetSearchTerm();
      }
    },
    [onResetSearchTerm]
  );

  return (
    <InputGroup>
      <Input
        placeholder={t('gallery.searchImages')}
        value={searchTerm}
        onChange={handleChangeInput}
        data-testid="image-search-input"
        onKeyDown={handleKeydown}
      />
      {isFetching && (
        <InputRightElement h="full" pe={2}>
          <Spinner size="sm" opacity={0.5} />
        </InputRightElement>
      )}
      {!isFetching && searchTerm.length && (
        <InputRightElement h="full" pe={2}>
          <IconButton
            onClick={onResetSearchTerm}
            size="sm"
            variant="link"
            aria-label={t('boards.clearSearch')}
            icon={<PiXBold />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
});

GallerySearch.displayName = 'GallerySearch';
