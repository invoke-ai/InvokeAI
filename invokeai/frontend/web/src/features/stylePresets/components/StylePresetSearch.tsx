import { IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { searchTermChanged, selectStylePresetSearchTerm } from 'features/stylePresets/store/stylePresetSlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const StylePresetSearch = () => {
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectStylePresetSearchTerm);
  const { t } = useTranslation();

  const handlePresetSearch = useCallback(
    (newSearchTerm: string) => {
      dispatch(searchTermChanged(newSearchTerm));
    },
    [dispatch]
  );

  const clearPresetSearch = useCallback(() => {
    dispatch(searchTermChanged(''));
  }, [dispatch]);

  const handleKeydown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        clearPresetSearch();
      }
    },
    [clearPresetSearch]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      handlePresetSearch(e.target.value);
    },
    [handlePresetSearch]
  );

  return (
    <InputGroup>
      <Input
        placeholder={t('stylePresets.searchByName')}
        value={searchTerm}
        onKeyDown={handleKeydown}
        onChange={handleChange}
      />
      {searchTerm && searchTerm.length && (
        <InputRightElement h="full" pe={2}>
          <IconButton
            onClick={clearPresetSearch}
            size="sm"
            variant="link"
            aria-label={t('boards.clearSearch')}
            icon={<PiXBold />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(StylePresetSearch);
