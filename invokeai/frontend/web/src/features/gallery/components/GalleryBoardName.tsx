import { ChevronUpIcon } from '@chakra-ui/icons';
import { Button, Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvText } from 'common/components/InvText/wrapper';
import { memo, useMemo } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';

const selector = createMemoizedSelector([stateSelector], (state) => {
  const { selectedBoardId } = state.gallery;

  return { selectedBoardId };
});

type Props = {
  isOpen: boolean;
  onToggle: () => void;
};

const GalleryBoardName = (props: Props) => {
  const { isOpen, onToggle } = props;
  const { selectedBoardId } = useAppSelector(selector);
  const boardName = useBoardName(selectedBoardId);

  const formattedBoardName = useMemo(() => {
    if (boardName.length > 20) {
      return `${boardName.substring(0, 20)}...`;
    }
    return boardName;
  }, [boardName]);

  return (
    <Flex
      as={Button}
      onClick={onToggle}
      size="sm"
      sx={{
        position: 'relative',
        gap: 2,
        w: 'full',
        justifyContent: 'space-between',
        alignItems: 'center',
        px: 2,
      }}
    >
      <InvText
        noOfLines={1}
        sx={{
          fontWeight: 'semibold',
          w: '100%',
          textAlign: 'center',
          color: 'base.200',
        }}
      >
        {formattedBoardName}
      </InvText>
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
