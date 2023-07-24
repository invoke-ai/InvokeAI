import { SelectItem } from '@mantine/core';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import {
  ESRGANModelName,
  esrganModelNameChanged,
} from 'features/parameters/store/postprocessingSlice';

export const ESRGAN_MODEL_NAMES: SelectItem[] = [
  {
    label: 'RealESRGAN x2 Plus',
    value: 'RealESRGAN_x2plus.pth',
    tooltip: 'Attempts to retain sharpness, low smoothing',
    group: 'x2 Upscalers',
  },
  {
    label: 'RealESRGAN x4 Plus',
    value: 'RealESRGAN_x4plus.pth',
    tooltip: 'Best for photos and highly detailed images, medium smoothing',
    group: 'x4 Upscalers',
  },
  {
    label: 'RealESRGAN x4 Plus (anime 6B)',
    value: 'RealESRGAN_x4plus_anime_6B.pth',
    tooltip: 'Best for anime/manga, high smoothing',
    group: 'x4 Upscalers',
  },
  {
    label: 'ESRGAN SRx4',
    value: 'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    tooltip: 'Retains sharpness, low smoothing',
    group: 'x4 Upscalers',
  },
];

export default function ParamESRGANModel() {
  const esrganModelName = useAppSelector(
    (state: RootState) => state.postprocessing.esrganModelName
  );

  const dispatch = useAppDispatch();

  const handleChange = (v: string) =>
    dispatch(esrganModelNameChanged(v as ESRGANModelName));

  return (
    <IAIMantineSelect
      label="ESRGAN Model"
      value={esrganModelName}
      itemComponent={IAIMantineSelectItemWithTooltip}
      onChange={handleChange}
      data={ESRGAN_MODEL_NAMES}
    />
  );
}
