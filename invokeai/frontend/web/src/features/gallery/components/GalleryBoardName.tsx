import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';

const GalleryBoardName = () => {
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
      my="1"
      justifyContent="center"
      fontSize="md"
      fontWeight="bold"
      borderWidth="thin"
      borderStyle="solid"
      borderRadius="base"
    >
      {formattedBoardName}
    </Flex>
  );
};

export default memo(GalleryBoardName);
