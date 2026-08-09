import type { ModelConfig } from '@features/models/core/types';

import { convertModelToDiffusers, deleteModel, reidentifyModel } from '@features/models/data/api';
import { removeModelsFromStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { removeModelsFromRelationships } from '@features/models/data/relationshipsStore';
import { pruneModelsUiKeys } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useCallback } from 'react';

/**
 * Lifecycle actions for a single model, shared by the detail page and the
 * library row context menu so behavior and notifications stay identical.
 * Confirmation UI is the caller's job; these just act and notify.
 */
type ModelActionTarget = Pick<ModelConfig, 'key' | 'name'>;

export const useModelActions = () => {
  const notify = useNotify();

  const remove = useCallback(
    async (model: ModelActionTarget) => {
      const owner = captureAccountScope();

      try {
        await deleteModel(model.key, owner.signal);

        assertAccountScopeCurrent(owner);
        removeModelsFromStore([model.key]);
        removeModelsFromRelationships([model.key]);
        pruneModelsUiKeys([model.key]);
        notify.success('Model deleted', model.name);
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error('Delete failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  const convert = useCallback(
    async (model: ModelActionTarget) => {
      const owner = captureAccountScope();

      try {
        const converted = await convertModelToDiffusers(model.key, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(converted);
        notify.success('Converted to diffusers', model.name);
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error('Conversion failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  const reidentify = useCallback(
    async (model: ModelActionTarget) => {
      const owner = captureAccountScope();

      try {
        const identified = await reidentifyModel(model.key, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(identified);
        notify.success('Model re-identified', model.name);
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        notify.error('Re-identify failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  return { convert, reidentify, remove };
};
