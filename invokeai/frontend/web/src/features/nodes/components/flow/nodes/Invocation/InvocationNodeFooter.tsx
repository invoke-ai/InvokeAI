import { Flex } from '@chakra-ui/react';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { memo } from 'react';
import EmbedWorkflowCheckbox from './EmbedWorkflowCheckbox';
import SaveToGalleryCheckbox from './SaveToGalleryCheckbox';
import UseCacheCheckbox from './UseCacheCheckbox';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { useFeatureStatus } from '../../../../../system/hooks/useFeatureStatus';

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
      sx={{
        w: 'full',
        borderBottomRadius: 'base',
        px: 2,
        py: 0,
        h: 6,
        justifyContent: 'space-between',
      }}
    >
      {isCacheEnabled && <UseCacheCheckbox nodeId={nodeId} />}
      {hasImageOutput && <EmbedWorkflowCheckbox nodeId={nodeId} />}
      {hasImageOutput && <SaveToGalleryCheckbox nodeId={nodeId} />}
    </Flex>
  );
};

export default memo(InvocationNodeFooter);
