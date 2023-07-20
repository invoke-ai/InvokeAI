import { ChevronUpIcon } from '@chakra-ui/icons';
import { Box, Button, Flex, Spacer, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { memo } from 'react';
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

  return (
    <Flex
      as={Button}
      onClick={onToggle}
      size="sm"
      variant="ghost"
      sx={{
        position: 'relative',
        gap: 2,
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
      <Spacer />
      <Box position="relative">
        <Text
          noOfLines={1}
          sx={{
            fontWeight: 600,
            color: 'base.800',
            _dark: {
              color: 'base.200',
            },
          }}
        >
          {boardName}
        </Text>
      </Box>
      <Spacer />
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
