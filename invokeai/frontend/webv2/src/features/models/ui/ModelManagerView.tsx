/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import { ensureModelsLoaded, getModelsSnapshot } from '@features/models/data/modelsStore';
import { InstallQueueBar } from '@features/models/ui/install-queue/InstallQueueBar';
import {
  closeModelDetail,
  openModelManagerTab,
  settleInitialModelsPane,
  useModelsUiSelector,
} from '@features/models/ui/uiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ManagerLayout } from '@platform/ui/ManagerLayout';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DetailPane } from './manager/DetailPane';
import { LibraryColumn } from './manager/LibraryColumn';

const openAddModels = () => openModelManagerTab('add');

/** Full model manager: the library, the tabbed detail pane, and the install queue under both. */
export const ModelManagerView = () => {
  const { t } = useTranslation();
  const detailOpen = useModelsUiSelector((snapshot) => snapshot.detailOpen);
  const queueFillsPane = useModelsUiSelector((snapshot) => snapshot.queueExpanded && snapshot.queueMaximized);
  const addAction = useMemo(() => ({ label: t('models.addModels'), onAdd: openAddModels }), [t]);

  useMountEffect(() => {
    const owner = captureAccountScope();

    // The starting pane is decided once the library has loaded, then left to the user.
    void ensureModelsLoaded()
      .catch(() => undefined)
      .then(() => {
        if (isAccountScopeCurrent(owner)) {
          const { models, status } = getModelsSnapshot();

          settleInitialModelsPane(status === 'loaded' && models.length === 0);
        }
      });
  });

  return (
    <ManagerLayout
      backLabel={t('models.backToList')}
      detail={<DetailPane />}
      footer={<InstallQueueBar />}
      footerFills={queueFillsPane}
      isDetailOpen={detailOpen ?? false}
      library={<LibraryColumn addAction={addAction} />}
      onBack={closeModelDetail}
    />
  );
};
