import { Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

const TopCenterPanel = () => {
  const name = useAppSelector((s) => s.workflow.name);

  return (
    <Text m={2} fontSize="lg" userSelect="none" noOfLines={1} wordBreak="break-all" fontWeight="semibold" opacity={0.8}>
      {name}
    </Text>
  );
};

export default memo(TopCenterPanel);
