import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectLastSelectedNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const NodeTemplateInspector = () => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const lastSelectedNode = selectLastSelectedNode(nodes);
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
