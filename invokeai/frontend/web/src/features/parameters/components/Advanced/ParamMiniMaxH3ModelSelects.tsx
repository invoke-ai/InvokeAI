import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  minimaxH3TransformerModelSelected,
  selectMiniMaxH3TransformerModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useMiniMaxH3CheckpointModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig } from 'services/api/types';

/**
 * MiniMax H3 Transformer (single file) Select
 *
 * Picks an optional single-file transformer checkpoint (e.g. the pruned int8 repack) that
 * replaces the diffusers folder's 62 GB bf16 transformer. The text encoder and both VAEs
 * still come from the folder install selected as the main model.
 */
const ParamMiniMaxH3TransformerSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useAppSelector(selectMiniMaxH3TransformerModel);
  const [modelConfigs, { isLoading }] = useMiniMaxH3CheckpointModels();

  const _onChange = useCallback(
    (model: MainModelConfig | null) => {
      if (model) {
        dispatch(minimaxH3TransformerModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(minimaxH3TransformerModelSelected(null));
      }
    },
    [dispatch]
  );

  const {
    options,
    value: comboValue,
    onChange,
    noOptionsMessage,
  } = useModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel: value,
    isLoading,
  });

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <FormLabel m={0}>{t('modelManager.minimaxH3TransformerModel')}</FormLabel>
      <Combobox
        value={comboValue}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.minimaxH3TransformerModelPlaceholder')}
      />
    </FormControl>
  );
});

ParamMiniMaxH3TransformerSelect.displayName = 'ParamMiniMaxH3TransformerSelect';

const ParamMiniMaxH3ModelSelects = () => {
  return <ParamMiniMaxH3TransformerSelect />;
};

export default memo(ParamMiniMaxH3ModelSelects);
