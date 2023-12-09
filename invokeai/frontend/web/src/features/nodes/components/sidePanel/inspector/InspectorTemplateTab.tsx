import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, ({ nodes }) => {
  const lastSelectedNodeId =
    nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find(
    (node) => node.id === lastSelectedNodeId
  );

  const lastSelectedNodeTemplate = lastSelectedNode
    ? nodes.nodeTemplates[lastSelectedNode.data.type]
    : undefined;

  return {
    template: lastSelectedNodeTemplate,
  };
});

const NodeTemplateInspector = () => {
  const { template } = useAppSelector(selector);
  const { t } = useTranslation();

  if (!template) {
    return (
      <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />
    );
  }

  return <DataViewer data={template} label={t('nodes.nodeTemplate')} />;
};

export default memo(NodeTemplateInspector);
