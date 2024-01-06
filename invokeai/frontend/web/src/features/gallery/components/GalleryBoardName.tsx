import { ChevronUpIcon } from '@chakra-ui/icons';
import { Button, Flex, Spacer } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';

type Props = {
  isOpen: boolean;
  onToggle: () => void;
};

const GalleryBoardName = (props: Props) => {
  const { isOpen, onToggle } = props;
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
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
      position="relative"
      gap={2}
      w="full"
      justifyContent="center"
      alignItems="center"
      px={2}
    >
      <Spacer />
      {formattedBoardName}
      <Spacer />
      <ChevronUpIcon
        transform={isOpen ? 'rotate(0deg)' : 'rotate(180deg)'}
        transitionProperty="common"
        transitionDuration="normal"
      />
    </Flex>
  );
};

export default memo(GalleryBoardName);
