import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterModel } from 'features/controlAdapters/hooks/useControlAdapterModel';
import { useControlAdapterModels } from 'features/controlAdapters/hooks/useControlAdapterModels';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import { controlAdapterModelChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToControlNetModelParam } from 'features/parameters/util/modelIdToControlNetModelParam';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlAdapterModelProps = {
  id: string;
};

const selector = createSelector(
  stateSelector,
  ({ generation }) => {
    const { model } = generation;
    return { mainModel: model };
  },
  defaultSelectorOptions
);

const ParamControlAdapterModel = ({ id }: ParamControlAdapterModelProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlAdapterType = useControlAdapterType(id);
  const model = useControlAdapterModel(id);
  const dispatch = useAppDispatch();

  const { mainModel } = useAppSelector(selector);
  const { t } = useTranslation();

  const models = useControlAdapterModels(controlAdapterType);

  const data = useMemo(() => {
    if (!models) {
      return [];
    }

    const data: {
      value: string;
      label: string;
      group: string;
      disabled: boolean;
      tooltip?: string;
    }[] = [];

    models.forEach((model) => {
      if (!model) {
        return;
      }

      const disabled = model?.base_model !== mainModel?.base_model;

      data.push({
        value: model.id,
        label: model.model_name,
        group: MODEL_TYPE_MAP[model.base_model],
        disabled,
        tooltip: disabled
          ? `${t('controlnet.incompatibleBaseModel')} ${model.base_model}`
          : undefined,
      });
    });

    data.sort((a, b) =>
      // sort 'none' to the top
      a.disabled ? 1 : b.disabled ? -1 : a.label.localeCompare(b.label)
    );

    return data;
  }, [mainModel?.base_model, models, t]);

  // grab the full model entity from the RTK Query cache
  const selectedModel = useMemo(
    () =>
      models.find(
        (m) =>
          m?.id ===
          `${model?.base_model}/${controlAdapterType}/${model?.model_name}`
      ),
    [controlAdapterType, model?.base_model, model?.model_name, models]
  );

  const handleModelChanged = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newControlNetModel = modelIdToControlNetModelParam(v);

      if (!newControlNetModel) {
        return;
      }

      dispatch(controlAdapterModelChanged({ id, model: newControlNetModel }));
    },
    [dispatch, id]
  );

  return (
    <IAIMantineSearchableSelect
      itemComponent={IAIMantineSelectItemWithTooltip}
      data={data}
      error={
        !selectedModel || mainModel?.base_model !== selectedModel.base_model
      }
      placeholder={t('controlnet.selectModel')}
      value={selectedModel?.id ?? null}
      onChange={handleModelChanged}
      disabled={!isEnabled}
      tooltip={selectedModel?.description}
    />
  );
};

export default memo(ParamControlAdapterModel);
