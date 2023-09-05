import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setMaskBlurMethod } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

type MaskBlurMethods = 'box' | 'gaussian';

const maskBlurMethods: SelectItem[] = [
  { label: 'Box Blur', value: 'box' },
  { label: 'Gaussian Blur', value: 'gaussian' },
];

export default function ParamMaskBlurMethod() {
  const maskBlurMethod = useAppSelector(
    (state: RootState) => state.generation.maskBlurMethod
  );
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleMaskBlurMethodChange = (v: string | null) => {
    if (!v) {
      return;
    }
    dispatch(setMaskBlurMethod(v as MaskBlurMethods));
  };

  return (
    <IAIMantineSelect
      value={maskBlurMethod}
      onChange={handleMaskBlurMethodChange}
      label={t('parameters.maskBlurMethod')}
      data={maskBlurMethods}
    />
  );
}
