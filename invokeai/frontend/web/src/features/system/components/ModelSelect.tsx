import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  modelSelected,
  setCurrentModelType,
} from 'features/parameters/store/generationSlice';

import { modelSelector } from '../store/modelSelectors';

export type ModelLoaderTypes = 'sd1_model_loader' | 'sd2_model_loader';

const MODEL_LOADER_MAP = {
  'sd-1': 'sd1_model_loader',
  'sd-2': 'sd2_model_loader',
};

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const {
    selectedModel,
    sd1PipelineModelDropDownData,
    sd2PipelineModelDropdownData,
  } = useAppSelector(modelSelector);

  useEffect(() => {
    if (selectedModel)
      dispatch(
        setCurrentModelType(
          MODEL_LOADER_MAP[selectedModel?.base_model] as ModelLoaderTypes
        )
      );
  }, [dispatch, selectedModel]);

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(modelSelected(v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.model')}
      value={selectedModel?.name ?? ''}
      placeholder="Pick one"
      data={sd1PipelineModelDropDownData.concat(sd2PipelineModelDropdownData)}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
