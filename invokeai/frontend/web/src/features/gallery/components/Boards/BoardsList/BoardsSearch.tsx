import { IconButton, Input, InputGroup, InputRightElement } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const BoardsSearch = () => {
  const dispatch = useAppDispatch();
  const boardSearchText = useAppSelector((s) => s.gallery.boardSearchText);
  const { t } = useTranslation();

  const handleBoardSearch = useCallback(
    (searchTerm: string) => {
      dispatch(boardSearchTextChanged(searchTerm));
    },
    [dispatch]
  );

  const clearBoardSearch = useCallback(() => {
    dispatch(boardSearchTextChanged(''));
  }, [dispatch]);

  const handleKeydown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      // exit search mode on escape
      if (e.key === 'Escape') {
        clearBoardSearch();
      }
    },
    [clearBoardSearch]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      handleBoardSearch(e.target.value);
    },
    [handleBoardSearch]
  );

  return (
    <InputGroup>
      <Input
        placeholder={t('boards.searchBoard')}
        value={boardSearchText}
        onKeyDown={handleKeydown}
        onChange={handleChange}
        data-testid="board-search-input"
      />
      {boardSearchText && boardSearchText.length && (
        <InputRightElement h="full" pe={2}>
          <IconButton
            onClick={clearBoardSearch}
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

export default memo(BoardsSearch);
