import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useModelCombobox } from 'common/hooks/useModelCombobox';
import {
  minimaxH3TextEncoderModelSelected,
  minimaxH3TransformerModelSelected,
  selectMiniMaxH3TextEncoderModel,
  selectMiniMaxH3TransformerModel,
} from 'features/controlLayers/store/paramsSlice';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useMiniMaxH3CheckpointModels, useMiniMaxH3TextEncoderModels } from 'services/api/hooks/modelsByType';
import type { MainModelConfig, Qwen3VLEncoderModelConfig } from 'services/api/types';

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

/**
 * MiniMax H3 Text Encoder (single file) Select
 *
 * Picks an optional single-file truncated Qwen3-VL-32B encoder (e.g. the int8 repack) that
 * replaces the diffusers folder's 62 GB bf16 text encoder. The tokenizer and processor still
 * come from the folder install selected as the main model.
 */
const ParamMiniMaxH3TextEncoderSelect = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const value = useAppSelector(selectMiniMaxH3TextEncoderModel);
  const [modelConfigs, { isLoading }] = useMiniMaxH3TextEncoderModels();

  const _onChange = useCallback(
    (model: Qwen3VLEncoderModelConfig | null) => {
      if (model) {
        dispatch(minimaxH3TextEncoderModelSelected(zModelIdentifierField.parse(model)));
      } else {
        dispatch(minimaxH3TextEncoderModelSelected(null));
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
      <FormLabel m={0}>{t('modelManager.minimaxH3TextEncoderModel')}</FormLabel>
      <Combobox
        value={comboValue}
        options={options}
        onChange={onChange}
        noOptionsMessage={noOptionsMessage}
        isClearable
        placeholder={t('modelManager.minimaxH3TextEncoderModelPlaceholder')}
      />
    </FormControl>
  );
});

ParamMiniMaxH3TextEncoderSelect.displayName = 'ParamMiniMaxH3TextEncoderSelect';

const ParamMiniMaxH3ModelSelects = () => {
  return (
    <>
      <ParamMiniMaxH3TransformerSelect />
      <ParamMiniMaxH3TextEncoderSelect />
    </>
  );
};

export default memo(ParamMiniMaxH3ModelSelects);
