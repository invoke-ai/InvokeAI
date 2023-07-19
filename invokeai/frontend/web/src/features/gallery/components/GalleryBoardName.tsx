import { ChevronUpIcon } from '@chakra-ui/icons';
import { Button, Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId } = state.gallery;

    return {
      selectedBoardId,
    };
  },
  defaultSelectorOptions
);

type Props = {
  isOpen: boolean;
  onToggle: () => void;
};

const GalleryBoardName = (props: Props) => {
  const { isOpen, onToggle } = props;
  const { selectedBoardId } = useAppSelector(selector);
  const { selectedBoardName } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => {
      let selectedBoardName = '';
      if (selectedBoardId === 'images') {
        selectedBoardName = 'All Images';
      } else if (selectedBoardId === 'assets') {
        selectedBoardName = 'All Assets';
      } else if (selectedBoardId === 'no_board') {
        selectedBoardName = 'No Board';
      } else if (selectedBoardId === 'batch') {
        selectedBoardName = 'Batch';
      } else {
        const selectedBoard = data?.find((b) => b.board_id === selectedBoardId);
        selectedBoardName = selectedBoard?.board_name || 'Unknown Board';
      }

      return { selectedBoardName };
    },
  });

  return (
    <Flex
      as={Button}
      onClick={onToggle}
      size="sm"
      variant="ghost"
      sx={{
        w: 'full',
        justifyContent: 'center',
        alignItems: 'center',
        px: 2,
        _hover: {
          bg: 'base.100',
          _dark: { bg: 'base.800' },
        },
      }}
    >
      <Text
        noOfLines={1}
        sx={{
          w: 'full',
          fontWeight: 600,
          color: 'base.800',
          _dark: {
            color: 'base.200',
          },
        }}
      >
        {selectedBoardName}
      </Text>
      <ChevronUpIcon
        sx={{
          transform: isOpen ? 'rotate(0deg)' : 'rotate(180deg)',
          transitionProperty: 'common',
          transitionDuration: 'normal',
        }}
      />
    </Flex>
  );
};

export default memo(GalleryBoardName);
