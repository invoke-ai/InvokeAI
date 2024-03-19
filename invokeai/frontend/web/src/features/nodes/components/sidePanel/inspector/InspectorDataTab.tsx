import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
const selector = createMemoizedSelector(selectNodesSlice, (nodes) => {
  const lastSelectedNodeId = nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find((node) => node.id === lastSelectedNodeId);

  return {
    data: lastSelectedNode?.data,
  };
});

const InspectorDataTab = () => {
  const { t } = useTranslation();
  const { data } = useAppSelector(selector);

  if (!data) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return <DataViewer data={data} label="Node Data" />;
};

export default memo(InspectorDataTab);
