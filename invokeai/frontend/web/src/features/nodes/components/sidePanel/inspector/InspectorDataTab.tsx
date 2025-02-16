import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { selectLastSelectedNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebounce } from 'use-debounce';

const selector = createSelector(selectNodesSlice, (nodes) => selectLastSelectedNode(nodes)?.data);

const InspectorDataTab = () => {
  const { t } = useTranslation();
  const lastSelectedNodeData = useAppSelector(selector);
  const [debouncedLastSelectedNodeData] = useDebounce(lastSelectedNodeData, 300);

  if (!debouncedLastSelectedNodeData) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return <DataViewer data={debouncedLastSelectedNodeData} label="Node Data" />;
};

export default memo(InspectorDataTab);
