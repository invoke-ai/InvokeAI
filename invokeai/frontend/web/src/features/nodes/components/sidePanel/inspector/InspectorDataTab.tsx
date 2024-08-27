import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { selectLastSelectedNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectNodesSlice, (nodes) => selectLastSelectedNode(nodes));

const InspectorDataTab = () => {
  const { t } = useTranslation();
  const lastSelectedNode = useAppSelector(selector);

  if (!lastSelectedNode) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return <DataViewer data={lastSelectedNode.data} label="Node Data" />;
};

export default memo(InspectorDataTab);
