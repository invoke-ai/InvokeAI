import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import type { CLIPVisionModelV2 } from 'features/controlLayers/store/types';
import { isCLIPVisionModelV2 } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

// at this time, ViT-L is the only supported clip model for FLUX IP adapter
const FLUX_CLIP_VISION = 'ViT-L';

const CLIP_VISION_OPTIONS = [
  { label: 'ViT-H', value: 'ViT-H' },
  { label: 'ViT-G', value: 'ViT-G' },
  { label: FLUX_CLIP_VISION, value: FLUX_CLIP_VISION },
];

type Props = {
  model: CLIPVisionModelV2;
  onChange: (clipVisionModel: CLIPVisionModelV2) => void;
};

export const CLIPVisionModel = memo(({ model, onChange }: Props) => {
  const { t } = useTranslation();

  const _onChangeCLIPVisionModel = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isCLIPVisionModelV2(v?.value));
      onChange(v.value);
    },
    [onChange]
  );

  const isFLUX = useAppSelector(selectIsFLUX);

  const clipVisionOptions = useMemo(() => {
    return CLIP_VISION_OPTIONS.map((option) => ({
      ...option,
      isDisabled: isFLUX && option.value !== FLUX_CLIP_VISION,
    }));
  }, [isFLUX]);

  const clipVisionModelValue = useMemo(() => {
    return CLIP_VISION_OPTIONS.find((o) => o.value === model);
  }, [model]);

  return (
    <FormControl width="max-content" minWidth={28}>
      <Combobox
        options={clipVisionOptions}
        placeholder={t('common.placeholderSelectAModel')}
        value={clipVisionModelValue}
        onChange={_onChangeCLIPVisionModel}
      />
    </FormControl>
  );
});

CLIPVisionModel.displayName = 'CLIPVisionModel';
