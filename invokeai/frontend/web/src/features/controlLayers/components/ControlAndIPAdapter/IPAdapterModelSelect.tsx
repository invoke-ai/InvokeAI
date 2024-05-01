import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import type { CLIPVisionModel } from 'features/controlLayers/util/controlAdapters';
import { isCLIPVisionModel } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useIPAdapterModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

const CLIP_VISION_OPTIONS = [
  { label: 'ViT-H', value: 'ViT-H' },
  { label: 'ViT-G', value: 'ViT-G' },
];

type Props = {
  modelKey: string | null;
  onChangeModel: (modelConfig: IPAdapterModelConfig) => void;
  clipVisionModel: CLIPVisionModel;
  onChangeCLIPVisionModel: (clipVisionModel: CLIPVisionModel) => void;
};

export const IPAdapterModelSelect = memo(
  ({ modelKey, onChangeModel, clipVisionModel, onChangeCLIPVisionModel }: Props) => {
    const { t } = useTranslation();
    const currentBaseModel = useAppSelector((s) => s.generation.model?.base);
    const [modelConfigs, { isLoading }] = useIPAdapterModels();
    const selectedModel = useMemo(() => modelConfigs.find((m) => m.key === modelKey), [modelConfigs, modelKey]);

    const _onChangeModel = useCallback(
      (modelConfig: IPAdapterModelConfig | null) => {
        if (!modelConfig) {
          return;
        }
        onChangeModel(modelConfig);
      },
      [onChangeModel]
    );

    const _onChangeCLIPVisionModel = useCallback<ComboboxOnChange>(
      (v) => {
        assert(isCLIPVisionModel(v?.value));
        onChangeCLIPVisionModel(v.value);
      },
      [onChangeCLIPVisionModel]
    );

    const getIsDisabled = useCallback(
      (model: AnyModelConfig): boolean => {
        const isCompatible = currentBaseModel === model.base;
        const hasMainModel = Boolean(currentBaseModel);
        return !hasMainModel || !isCompatible;
      },
      [currentBaseModel]
    );

    const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
      modelConfigs,
      onChange: _onChangeModel,
      selectedModel,
      getIsDisabled,
      isLoading,
    });

    const clipVisionModelValue = useMemo(
      () => CLIP_VISION_OPTIONS.find((o) => o.value === clipVisionModel),
      [clipVisionModel]
    );

    return (
      <Flex gap={4}>
        <Tooltip label={selectedModel?.description}>
          <FormControl isInvalid={!value || currentBaseModel !== selectedModel?.base} w="full">
            <Combobox
              options={options}
              placeholder={t('controlnet.selectModel')}
              value={value}
              onChange={onChange}
              noOptionsMessage={noOptionsMessage}
            />
          </FormControl>
        </Tooltip>
        {selectedModel?.format === 'checkpoint' && (
          <FormControl isInvalid={!value || currentBaseModel !== selectedModel?.base} width="max-content" minWidth={28}>
            <Combobox
              options={CLIP_VISION_OPTIONS}
              placeholder={t('controlnet.selectCLIPVisionModel')}
              value={clipVisionModelValue}
              onChange={_onChangeCLIPVisionModel}
            />
          </FormControl>
        )}
      </Flex>
    );
  }
);

IPAdapterModelSelect.displayName = 'IPAdapterModelSelect';
