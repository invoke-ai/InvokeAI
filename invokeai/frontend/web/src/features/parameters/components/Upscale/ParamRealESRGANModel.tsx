import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { GroupBase } from 'chakra-react-select';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import {
  esrganModelNameChanged,
  isParamESRGANModelName,
} from 'features/parameters/store/postprocessingSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: GroupBase<InvSelectOption>[] = [
  {
    label: 'x2 Upscalers',
    options: [
      {
        label: 'RealESRGAN x2 Plus',
        value: 'RealESRGAN_x2plus.pth',
        description: 'Attempts to retain sharpness, low smoothing',
      },
    ],
  },
  {
    label: 'x4 Upscalers',
    options: [
      {
        label: 'RealESRGAN x4 Plus',
        value: 'RealESRGAN_x4plus.pth',
        description:
          'Best for photos and highly detailed images, medium smoothing',
      },
      {
        label: 'RealESRGAN x4 Plus (anime 6B)',
        value: 'RealESRGAN_x4plus_anime_6B.pth',
        description: 'Best for anime/manga, high smoothing',
      },
      {
        label: 'ESRGAN SRx4',
        value: 'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
        description: 'Retains sharpness, low smoothing',
      },
    ],
  },
];

export default function ParamESRGANModel() {
  const { t } = useTranslation();

  const esrganModelName = useAppSelector(
    (state: RootState) => state.postprocessing.esrganModelName
  );

  const dispatch = useAppDispatch();

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isParamESRGANModelName(v?.value)) {
        return;
      }
      dispatch(esrganModelNameChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () =>
      options
        .flatMap((o) => o.options)
        .find((m) => m.value === esrganModelName),
    [esrganModelName]
  );

  return (
    <InvControl label={t('models.esrganModel')}>
      <InvSelect value={value} onChange={onChange} options={options} />
    </InvControl>
  );
}
