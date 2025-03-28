import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { AddNodeCmdk } from 'features/nodes/components/flow/AddNodeCmdk/AddNodeCmdk';
import { TopCenterPanel } from 'features/nodes/components/flow/panels/TopPanel/TopCenterPanel';
import { TopLeftPanel } from 'features/nodes/components/flow/panels/TopPanel/TopLeftPanel';
import { TopRightPanel } from 'features/nodes/components/flow/panels/TopPanel/TopRightPanel';
import WorkflowEditorSettings from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import { useGetOpenAPISchemaQuery } from 'services/api/endpoints/appInfo';

import { Flow } from './flow/Flow';
import BottomLeftPanel from './flow/panels/BottomLeftPanel/BottomLeftPanel';
import MinimapPanel from './flow/panels/MinimapPanel/MinimapPanel';

const FOCUS_REGION_STYLES: SystemStyleObject = {
  display: 'flex',
  position: 'relative',
  width: 'full',
  height: 'full',
  alignItems: 'center',
  justifyContent: 'center',
};

const NodeEditor = () => {
  const { data, isLoading } = useGetOpenAPISchemaQuery();
  const { t } = useTranslation();

  return (
    <FocusRegionWrapper region="workflows" layerStyle="first" sx={FOCUS_REGION_STYLES}>
      {data && (
        <>
          <Flow />
          <AddNodeCmdk />
          <TopLeftPanel />
          <TopCenterPanel />
          <TopRightPanel />
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
