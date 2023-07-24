import { ChevronUpIcon } from '@chakra-ui/icons';
import { Button, Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo, useMemo } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId } = state.gallery;

    return { selectedBoardId };
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
  const boardName = useBoardName(selectedBoardId);
  // const { totalImages, totalAssets } = useBoardTotal(selectedBoardId);

  const formattedBoardName = useMemo(() => {
    if (boardName.length > 20) {
      return `${boardName.substring(0, 20)}...`;
    }
    return boardName;
    // if (!boardName) {
    //   return '';
    // }

    // if (boardName && (totalImages === undefined || totalAssets === undefined)) {
    //   return boardName;
    // }

    // const count = `${totalImages}/${totalAssets}`;

    // if (boardName.length > 20) {
    //   return `${boardName.substring(0, 20)}... (${count})`;
    // }
    // return `${boardName} (${count})`;
  }, [boardName]);

  return (
    <Flex
      as={Button}
      onClick={onToggle}
      size="sm"
      // variant="ghost"
      sx={{
        position: 'relative',
        gap: 2,
        w: 'full',
        justifyContent: 'space-between',
        alignItems: 'center',
        px: 2,
        // bg: 'base.100',
        // _dark: { bg: 'base.800' },
        // _hover: {
        //   bg: 'base.200',
        //   _dark: { bg: 'base.700' },
        // },
      }}
    >
      <Text
        noOfLines={1}
        sx={{
          fontWeight: 600,
          w: '100%',
          textAlign: 'center',
          color: 'base.800',
          _dark: {
            color: 'base.200',
          },
        }}
      >
        {formattedBoardName}
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
