import 'reactflow/dist/style.css';

import { Flex } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AddNodeCmdk } from 'features/nodes/components/flow/AddNodeCmdk/AddNodeCmdk';
import TopPanel from 'features/nodes/components/flow/panels/TopPanel/TopPanel';
import { LoadWorkflowFromGraphModal } from 'features/workflowLibrary/components/LoadWorkflowFromGraphModal/LoadWorkflowFromGraphModal';
import { SaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog/SaveWorkflowAsDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { MdDeviceHub } from 'react-icons/md';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { Flow } from './flow/Flow';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const NodeEditor = () => {
  const { data, isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();
  return (
    <Flex
      layerStyle="first"
      position="relative"
      width="full"
      height="full"
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
    >
      {data && (
        <>
          <Flow />
          <AddNodeCmdk />
          <TopPanel />
          <BottomLeftPanel />
          <MinimapPanel />
          <SaveWorkflowAsDialog />
          <LoadWorkflowFromGraphModal />
        </>
      )}
      {isLoading && <IAINoContentFallback label={t('nodes.loadingNodes')} icon={MdDeviceHub} />}
    </Flex>
  );
};

export default memo(NodeEditor);
