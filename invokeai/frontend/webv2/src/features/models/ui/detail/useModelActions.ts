import type { ModelConfig } from '@features/models/core/types';

import { convertModelToDiffusers, deleteModel, reidentifyModel } from '@features/models/data/api';
import { removeModelsFromStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { removeModelsFromRelationships } from '@features/models/data/relationshipsStore';
import { refreshStartersIfLoaded } from '@features/models/data/startersStore';
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
  // Separate instances: `run` ignores calls while its own action is in
  // flight, so sharing one would let a slow convert swallow a remove.
  // The busy flags stay unused.
  const { run: runRemove } = useScopedAction();
  const { run: runConvert } = useScopedAction();
  const { run: runReidentify } = useScopedAction();

  const remove = useCallback(
    (model: ModelActionTarget) =>
      runRemove(
        async (owner) => {
          await deleteModel(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          removeModelsFromStore([model.key]);
          removeModelsFromRelationships([model.key]);
          pruneModelsUiKeys([model.key]);
          // A deleted starter must lose its "Installed" badge in Add Models.
          refreshStartersIfLoaded();
          notify.success(t('models.modelDeleted'), model.name);
        },
        (message) => notify.error(t('models.deleteFailed'), message)
      ),
    [notify, runRemove, t]
  );

  const convert = useCallback(
    (model: ModelActionTarget) =>
      runConvert(
        async (owner) => {
          const converted = await convertModelToDiffusers(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(converted);
          notify.success(t('models.convertedToDiffusers'), model.name);
        },
        (message) => notify.error(t('models.conversionFailed'), message)
      ),
    [notify, runConvert, t]
  );

  const reidentify = useCallback(
    (model: ModelActionTarget) =>
      runReidentify(
        async (owner) => {
          const identified = await reidentifyModel(model.key, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(identified);
          notify.success(t('models.modelReidentified'), model.name);
        },
        (message) => notify.error(t('models.reidentifyFailed'), message)
      ),
    [notify, runReidentify, t]
  );

  return { convert, reidentify, remove };
};
