/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import { NodePackList } from '@features/nodes/ui/library/NodePackList';
import { openNodePackDetail, updateNodesUi, useNodesUiSelector } from '@features/nodes/ui/nodesUiStore';
import { ManagerColumn } from '@platform/ui/ManagerLayout';
import { useTranslation } from 'react-i18next';

import { ReloadNodesButton } from './ReloadNodesButton';

/** Persistent custom-node pack list, matching the model manager's library column. */
export const LibraryColumn = () => {
  const { t } = useTranslation();
  const activePackName = useNodesUiSelector((snapshot) => snapshot.activePackName);
  const filters = useNodesUiSelector((snapshot) => snapshot.filters);
  const error = useCustomNodesSelector((snapshot) => snapshot.error);
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const status = useCustomNodesSelector((snapshot) => snapshot.status);

  return (
    <ManagerColumn actions={<ReloadNodesButton />} count={nodePacks.length} title={t('nodes.nodePacks')}>
      <NodePackList
        activePackName={activePackName}
        error={error}
        filters={filters}
        packs={nodePacks}
        status={status}
        onFiltersChange={(next) => updateNodesUi({ filters: next })}
        onSelect={openNodePackDetail}
        onUninstalled={(packName) => {
          if (activePackName === packName) {
            updateNodesUi({ activePackName: null });
          }
        }}
      />
    </ManagerColumn>
  );
};
