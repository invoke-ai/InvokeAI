/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import { ensureCustomNodePacksLoaded, getCustomNodesSnapshot } from '@features/nodes/data/nodesStore';
import { NodeActivityBar } from '@features/nodes/ui/activity/NodeActivityBar';
import {
  closeNodePackDetail,
  openNodesManagerTab,
  settleInitialNodesPane,
  useNodesUiSelector,
} from '@features/nodes/ui/nodesUiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ManagerLayout } from '@platform/ui/ManagerLayout';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DetailPane } from './manager/DetailPane';
import { LibraryColumn } from './manager/LibraryColumn';

const openAddNodes = () => openNodesManagerTab('add');

/** Full custom nodes manager: the pack library, the detail pane, and recent activity under both. */
export const NodeManagerView = () => {
  const { t } = useTranslation();
  const detailOpen = useNodesUiSelector((snapshot) => snapshot.detailOpen);
  const addAction = useMemo(() => ({ label: t('nodes.addNodes'), onAdd: openAddNodes }), [t]);

  useMountEffect(() => {
    const owner = captureAccountScope();

    // The starting pane is decided once the library has loaded, then left to the user.
    void ensureCustomNodePacksLoaded()
      .catch(() => undefined)
      .then(() => {
        if (isAccountScopeCurrent(owner)) {
          const { nodePacks, status } = getCustomNodesSnapshot();

          settleInitialNodesPane(status === 'loaded' && nodePacks.length === 0);
        }
      });
  });

  return (
    <ManagerLayout
      backLabel={t('nodes.backToList')}
      detail={<DetailPane />}
      footer={<NodeActivityBar />}
      isDetailOpen={detailOpen ?? false}
      library={<LibraryColumn addAction={addAction} />}
      onBack={closeNodePackDetail}
    />
  );
};
