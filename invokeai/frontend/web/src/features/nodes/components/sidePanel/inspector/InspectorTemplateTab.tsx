import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { $templates, selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const NodeTemplateInspector = () => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const lastSelectedNodeId = nodes.selectedNodes[nodes.selectedNodes.length - 1];
        const lastSelectedNode = nodes.nodes.find((node) => node.id === lastSelectedNodeId);
        const lastSelectedNodeTemplate = lastSelectedNode ? templates[lastSelectedNode.data.type] : undefined;

        return lastSelectedNodeTemplate;
      }),
    [templates]
  );
  const template = useAppSelector(selector);
  const { t } = useTranslation();

  if (!template) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return <DataViewer data={template} label={t('nodes.nodeTemplate')} />;
};

export default memo(NodeTemplateInspector);
