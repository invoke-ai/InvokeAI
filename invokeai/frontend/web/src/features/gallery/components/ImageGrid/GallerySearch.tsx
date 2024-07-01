import { IconButton, Input, InputGroup, InputRightElement, Spinner } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesQueryArgs } from 'features/gallery/store/gallerySelectors';
import { searchTermChanged } from 'features/gallery/store/gallerySlice';
import { debounce } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useListImagesQuery } from 'services/api/endpoints/images';

export const GallerySearch = () => {
  const dispatch = useAppDispatch();
  const { searchTerm } = useAppSelector((s) => s.gallery);
  const { t } = useTranslation();

  const queryArgs = useAppSelector(selectListImagesQueryArgs);
  const { isFetching } = useListImagesQuery(queryArgs);

  const [searchTermInput, setSearchTermInput] = useState(searchTerm);

  const debouncedSetSearchTerm = useMemo(() => {
    return debounce((value: string) => {
      dispatch(searchTermChanged(value));
    }, 1000);
  }, [dispatch]);

  const handleChangeInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setSearchTermInput(e.target.value);
      debouncedSetSearchTerm(e.target.value);
    },
    [debouncedSetSearchTerm]
  );

  const handleClearInput = useCallback(() => {
    setSearchTermInput('');
    dispatch(searchTermChanged(''));
  }, [dispatch]);

  return (
    <InputGroup>
      <Input
        placeholder={t('gallery.searchImages')}
        value={searchTermInput}
        onChange={handleChangeInput}
        data-testid="image-search-input"
        disabled={isFetching}
      />
      {isFetching ? (
        <InputRightElement h="full" pe={2}>
          <Spinner size="sm" />
        </InputRightElement>
      ) : (
        searchTermInput.length && (
          <InputRightElement h="full" pe={2}>
            <IconButton
              onClick={handleClearInput}
              size="sm"
              variant="link"
              aria-label={t('boards.clearSearch')}
              icon={<PiXBold />}
            />
          </InputRightElement>
        )
      )}
    </InputGroup>
  );
};
