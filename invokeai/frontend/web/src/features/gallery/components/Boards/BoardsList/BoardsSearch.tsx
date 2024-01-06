// import { CloseIcon } from '@chakra-ui/icons';
import { Input, InputGroup, InputRightElement } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi'

const selector = createMemoizedSelector([stateSelector], ({ gallery }) => {
  const { boardSearchText } = gallery;
  return { boardSearchText };
});

const BoardsSearch = () => {
  const dispatch = useAppDispatch();
  const { boardSearchText } = useAppSelector(selector);
  const inputRef = useRef<HTMLInputElement>(null);
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

  useEffect(() => {
    // focus the search box on mount
    if (!inputRef.current) {
      return;
    }
    inputRef.current.focus();
  }, []);

  return (
    <InputGroup>
      <Input
        ref={inputRef}
        placeholder={t('boards.searchBoard')}
        value={boardSearchText}
        onKeyDown={handleKeydown}
        onChange={handleChange}
        data-testid="board-search-input"
      />
      {boardSearchText && boardSearchText.length && (
        <InputRightElement h="full" pe={2}>
          <InvIconButton
            onClick={clearBoardSearch}
            size="sm"
            variant="ghost"
            aria-label={t('boards.clearSearch')}
            icon={<PiXBold />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(BoardsSearch);
