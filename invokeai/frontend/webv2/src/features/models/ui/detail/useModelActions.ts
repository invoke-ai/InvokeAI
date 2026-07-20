import type { ModelConfig } from '@features/models/core/types';

import { convertModelToDiffusers, deleteModel, reidentifyModel } from '@features/models/data/api';
import { removeModelsFromStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { pruneModelsUiKeys } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
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
      try {
        await deleteModel(model.key);
        removeModelsFromStore([model.key]);
        pruneModelsUiKeys([model.key]);
        notify.success('Model deleted', model.name);
      } catch (error) {
        notify.error('Delete failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  const convert = useCallback(
    async (model: ModelActionTarget) => {
      try {
        replaceModelInStore(await convertModelToDiffusers(model.key));
        notify.success('Converted to diffusers', model.name);
      } catch (error) {
        notify.error('Conversion failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  const reidentify = useCallback(
    async (model: ModelActionTarget) => {
      try {
        replaceModelInStore(await reidentifyModel(model.key));
        notify.success('Model re-identified', model.name);
      } catch (error) {
        notify.error('Re-identify failed', error instanceof Error ? error.message : String(error));
      }
    },
    [notify]
  );

  return { convert, reidentify, remove };
};
