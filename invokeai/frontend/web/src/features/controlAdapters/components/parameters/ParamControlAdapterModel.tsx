import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterModel } from 'features/controlAdapters/hooks/useControlAdapterModel';
import { useControlAdapterModelEntities } from 'features/controlAdapters/hooks/useControlAdapterModelEntities';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import { controlAdapterModelChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { pick } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  ControlNetModelConfigEntity,
  IPAdapterModelConfigEntity,
  T2IAdapterModelConfigEntity,
} from 'services/api/endpoints/models';

type ParamControlAdapterModelProps = {
  id: string;
};

const selector = createMemoizedSelector(stateSelector, ({ generation }) => {
  const { model } = generation;
  return { mainModel: model };
});

const ParamControlAdapterModel = ({ id }: ParamControlAdapterModelProps) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const controlAdapterType = useControlAdapterType(id);
  const model = useControlAdapterModel(id);
  const dispatch = useAppDispatch();

  const { mainModel } = useAppSelector(selector);
  const { t } = useTranslation();

  const models = useControlAdapterModelEntities(controlAdapterType);

  const _onChange = useCallback(
    (
      model:
        | ControlNetModelConfigEntity
        | IPAdapterModelConfigEntity
        | T2IAdapterModelConfigEntity
        | null
    ) => {
      if (!model) {
        return;
      }
      dispatch(
        controlAdapterModelChanged({
          id,
          model: pick(model, 'base_model', 'model_name'),
        })
      );
    },
    [dispatch, id]
  );

  const { options, value, onChange, noOptionsMessage } =
    useGroupedModelInvSelect({
      modelEntities: models,
      onChange: _onChange,
      selectedModel:
        model && controlAdapterType
          ? { ...model, model_type: controlAdapterType }
          : null,
    });

  return (
    <InvTooltip label={value?.description}>
      <InvControl
        isDisabled={!isEnabled}
        isInvalid={!value || mainModel?.base_model !== model?.base_model}
      >
        <InvSelect
          options={options}
          placeholder={t('controlnet.selectModel')}
          value={value}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </InvControl>
    </InvTooltip>
  );
};

export default memo(ParamControlAdapterModel);
