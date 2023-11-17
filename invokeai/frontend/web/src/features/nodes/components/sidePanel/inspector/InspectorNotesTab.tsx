import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { isInvocationNode } from 'features/nodes/types/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import NotesTextarea from '../../flow/nodes/Invocation/NotesTextarea';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const lastSelectedNodeId =
      nodes.selectedNodes[nodes.selectedNodes.length - 1];

    const lastSelectedNode = nodes.nodes.find(
      (node) => node.id === lastSelectedNodeId
    );

    if (!isInvocationNode(lastSelectedNode)) {
      return;
    }

    return lastSelectedNode.id;
  },
  defaultSelectorOptions
);

const InspectorNotesTab = () => {
  const nodeId = useAppSelector(selector);
  const { t } = useTranslation();

  if (!nodeId) {
    return (
      <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />
    );
  }

  return <NotesTextarea nodeId={nodeId} />;
};

export default memo(InspectorNotesTab);
