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
import { memo } from 'react';

const selector = createSelector(
  [stateSelector],
  ({ boards }) => {
    const { searchText } = boards;
    return { searchText };
  },
  defaultSelectorOptions
);

type Props = {
  setSearchMode: (searchMode: boolean) => void;
};

const BoardsSearch = (props: Props) => {
  const { setSearchMode } = props;
  const dispatch = useAppDispatch();
  const { searchText } = useAppSelector(selector);

  const handleBoardSearch = (searchTerm: string) => {
    setSearchMode(searchTerm.length > 0);
    dispatch(setBoardSearchText(searchTerm));
  };
  const clearBoardSearch = () => {
    setSearchMode(false);
    dispatch(setBoardSearchText(''));
  };

  return (
    <InputGroup>
      <Input
        placeholder="Search Boards..."
        value={searchText}
        onChange={(e) => {
          handleBoardSearch(e.target.value);
        }}
      />
      {searchText && searchText.length && (
        <InputRightElement>
          <IconButton
            onClick={clearBoardSearch}
            size="xs"
            variant="ghost"
            aria-label="Clear Search"
            icon={<CloseIcon boxSize={3} />}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
};

export default memo(BoardsSearch);
