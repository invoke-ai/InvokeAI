import 'reactflow/dist/style.css';

import { Flex } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useFocusRegion } from 'common/hooks/focus';
import { AddNodeCmdk } from 'features/nodes/components/flow/AddNodeCmdk/AddNodeCmdk';
import TopPanel from 'features/nodes/components/flow/panels/TopPanel/TopPanel';
import WorkflowEditorSettings from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { LoadWorkflowFromGraphModal } from 'features/workflowLibrary/components/LoadWorkflowFromGraphModal/LoadWorkflowFromGraphModal';
import { SaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog/SaveWorkflowAsDialog';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { Flow } from './flow/Flow';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const NodeEditor = () => {
  const { data, isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  useFocusRegion('workflows', ref);

  return (
    <Flex
      tabIndex={-1}
      ref={ref}
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
      <WorkflowEditorSettings />
      {isLoading && <IAINoContentFallback label={t('nodes.loadingNodes')} icon={PiFlowArrowBold} />}
    </Flex>
  );
};

export default memo(NodeEditor);
