import { Text } from '@chakra-ui/layout';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

const TopCenterPanel = () => {
  const name = useAppSelector(
    (state) => state.workflow.name || 'Untitled Workflow'
  );

  return (
    <Text
      m={2}
      fontSize="lg"
      userSelect="none"
      noOfLines={1}
      wordBreak="break-all"
      fontWeight={600}
      opacity={0.8}
    >
      {name}
    </Text>
  );
};

export default memo(TopCenterPanel);
