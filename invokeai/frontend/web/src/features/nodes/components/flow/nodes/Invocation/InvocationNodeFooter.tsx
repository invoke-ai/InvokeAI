import { Flex } from '@chakra-ui/react';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';

import SaveToGalleryCheckbox from './SaveToGalleryCheckbox';
import UseCacheCheckbox from './UseCacheCheckbox';

type Props = {
  nodeId: string;
};

const InvocationNodeFooter = ({ nodeId }: Props) => {
  const hasImageOutput = useHasImageOutput(nodeId);
  const isCacheEnabled = useFeatureStatus('invocationCache').isFeatureEnabled;
  return (
    <Flex
      className={DRAG_HANDLE_CLASSNAME}
      layerStyle="nodeFooter"
      w="full"
      borderBottomRadius="base"
      px={2}
      py={0}
      h={8}
      justifyContent="space-between"
    >
      {isCacheEnabled && <UseCacheCheckbox nodeId={nodeId} />}
      {hasImageOutput && <SaveToGalleryCheckbox nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(InvocationNodeFooter);
