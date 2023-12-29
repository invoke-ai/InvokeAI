import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import { useGroupedModelInvSelect } from 'common/components/InvSelect/useGroupedModelInvSelect';
import { loraAdded } from 'features/lora/store/loraSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { useGetLoRAModelsQuery } from 'services/api/endpoints/models';

const selector = createMemoizedSelector(stateSelector, ({ lora }) => ({
  addedLoRAs: lora.loras,
}));

const LoRASelect = () => {
  const dispatch = useAppDispatch();
  const { data, isLoading } = useGetLoRAModelsQuery();
  const { t } = useTranslation();
  const { addedLoRAs } = useAppSelector(selector);
  const currentBaseModel = useAppSelector(
    (state) => state.generation.model?.base_model
  );

  const getIsDisabled = (lora: LoRAModelConfigEntity): boolean => {
    const isCompatible = currentBaseModel === lora.base_model;
    const isAdded = Boolean(addedLoRAs[lora.id]);
    const hasMainModel = Boolean(currentBaseModel);
    return !hasMainModel || !isCompatible || isAdded;
  };

  const _onChange = useCallback(
    (lora: LoRAModelConfigEntity | null) => {
      if (!lora) {
        return;
      }
      dispatch(loraAdded(lora));
    },
    [dispatch]
  );

  const { options, onChange } = useGroupedModelInvSelect({
    modelEntities: data,
    getIsDisabled,
    onChange: _onChange,
  });

  const placeholder = useMemo(() => {
    if (isLoading) {
      return t('common.loading');
    }

    if (options.length === 0) {
      return t('models.noLoRAsInstalled');
    }

    return t('models.addLora');
  }, [isLoading, options.length, t]);

  const noOptionsMessage = useCallback(() => t('models.noMatchingLoRAs'), [t]);

  return (
    <InvControl label={t('models.lora')} isDisabled={!options.length}>
      <InvSelect
        placeholder={placeholder}
        value={null}
        options={options}
        noOptionsMessage={noOptionsMessage}
        onChange={onChange}
        data-testid="add-lora"
        w="full"
      />
    </InvControl>
  );
};

export default memo(LoRASelect);
