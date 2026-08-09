import type { ModelConfig } from '@features/models/core/types';

import { convertModelToDiffusers, deleteModel, reidentifyModel } from '@features/models/data/api';
import { removeModelsFromStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { removeModelsFromRelationships } from '@features/models/data/relationshipsStore';
import { useScopedAction } from '@features/models/ui/shared/useScopedAction';
import { pruneModelsUiKeys } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Lifecycle actions for a single model, shared by the detail page and the
 * library row context menu so behavior and notifications stay identical.
 * Confirmation UI is the caller's job; these just act and notify.
 */
type ModelActionTarget = Pick<ModelConfig, 'key' | 'name'>;

export const useModelActions = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  // None of these actions surface a busy flag, so one shared instance's `run`
  // covers all three and its isBusy is simply ignored.
  const { run } = useScopedAction();

  const remove = useCallback(
    (model: ModelActionTarget) =>
      run(
        async (owner) => {
          await deleteModel(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          removeModelsFromStore([model.key]);
          removeModelsFromRelationships([model.key]);
          pruneModelsUiKeys([model.key]);
          notify.success(t('models.modelDeleted'), model.name);
        },
        (message) => notify.error(t('models.deleteFailed'), message)
      ),
    [notify, run, t]
  );

  const convert = useCallback(
    (model: ModelActionTarget) =>
      run(
        async (owner) => {
          const converted = await convertModelToDiffusers(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(converted);
          notify.success(t('models.convertedToDiffusers'), model.name);
        },
        (message) => notify.error(t('models.conversionFailed'), message)
      ),
    [notify, run, t]
  );

  const reidentify = useCallback(
    (model: ModelActionTarget) =>
      run(
        async (owner) => {
          const identified = await reidentifyModel(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(identified);
          notify.success(t('models.modelReidentified'), model.name);
        },
        (message) => notify.error(t('models.reidentifyFailed'), message)
      ),
    [notify, run, t]
  );

  return { convert, reidentify, remove };
};
