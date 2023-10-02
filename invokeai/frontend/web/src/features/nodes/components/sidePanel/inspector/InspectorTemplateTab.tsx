import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const lastSelectedNodeId =
      nodes.present.selectedNodes[nodes.present.selectedNodes.length - 1];

    const lastSelectedNode = nodes.present.nodes.find(
      (node) => node.id === lastSelectedNodeId
    );

    const lastSelectedNodeTemplate = lastSelectedNode
      ? nodes.present.nodeTemplates[lastSelectedNode.data.type]
      : undefined;

    return {
      template: lastSelectedNodeTemplate,
    };
  },
  defaultSelectorOptions
);

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
