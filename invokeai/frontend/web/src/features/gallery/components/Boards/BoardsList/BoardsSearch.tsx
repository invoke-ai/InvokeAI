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
import { setBoardSearchText } from 'features/gallery/store/boardSlice';
import {
  ChangeEvent,
  KeyboardEvent,
  memo,
  useCallback,
  useEffect,
  useRef,
} from 'react';

const selector = createSelector(
  [stateSelector],
  ({ boards }) => {
    const { searchText } = boards;
    return { searchText };
  },
  defaultSelectorOptions
);

type Props = {
  setIsSearching?: (isSearching: boolean) => void;
};

const BoardsSearch = (props: Props) => {
  const { setIsSearching } = props;
  const dispatch = useAppDispatch();
  const { searchText } = useAppSelector(selector);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleBoardSearch = useCallback(
    (searchTerm: string) => {
      dispatch(setBoardSearchText(searchTerm));
    },
    [dispatch]
  );

  const clearBoardSearch = useCallback(() => {
    dispatch(setBoardSearchText(''));
    setIsSearching && setIsSearching(false);
  }, [dispatch, setIsSearching]);

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
        placeholder="Search Boards..."
        value={searchText}
        onKeyDown={handleKeydown}
        onChange={handleChange}
      />
      {searchText && searchText.length && (
        <InputRightElement>
          <IconButton
            onClick={clearBoardSearch}
            size="xs"
            variant="ghost"
            aria-label="Clear Search"
            opacity={0.5}
            icon={<CloseIcon boxSize={2} />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(BoardsSearch);
