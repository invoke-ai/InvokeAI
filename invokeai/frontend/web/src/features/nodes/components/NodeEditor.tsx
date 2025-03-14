import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AddNodeCmdk } from 'features/nodes/components/flow/AddNodeCmdk/AddNodeCmdk';
import TopPanel from 'features/nodes/components/flow/panels/TopPanel/TopPanel';
import WorkflowEditorSettings from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { Flow } from './flow/Flow';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const NodeEditor = () => {
  const { data, isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();

  return (
    <FocusRegionWrapper
      region="workflows"
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
        </>
      )}
      <WorkflowEditorSettings />
      {isLoading && <IAINoContentFallback label={t('nodes.loadingNodes')} icon={PiFlowArrowBold} />}
    </FocusRegionWrapper>
  );
};

export default memo(NodeEditor);
