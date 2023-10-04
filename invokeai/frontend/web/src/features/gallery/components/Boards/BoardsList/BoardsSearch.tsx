import { CloseIcon } from '@chakra-ui/icons';
import {
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import {
  ChangeEvent,
  KeyboardEvent,
  memo,
  useCallback,
  useEffect,
  useRef,
} from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  ({ gallery }) => {
    const { boardSearchText } = gallery;
    return { boardSearchText };
  },
  defaultSelectorOptions
);

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
        <InputRightElement>
          <IconButton
            onClick={clearBoardSearch}
            size="xs"
            variant="ghost"
            aria-label={t('boards.clearSearch')}
            opacity={0.5}
            icon={<CloseIcon boxSize={2} />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(BoardsSearch);
