import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSearchableSelect';
import { useCallback } from 'react';
import { setSDXLStylePreset } from '../store/sdxlSlice';

const sdxlPresetData: IAISelectDataType[] = [
  { label: 'None', value: 'none' },
  { label: 'Enhance', value: 'enhance' },
  { label: 'Anime Style', value: 'anime_style' },
  { label: 'Photo Realistic', value: 'photo_realistic' },
  { label: 'Digital Art', value: 'digital_art' },
  { label: 'Comic Book', value: 'comic_book' },
  { label: 'Fantasy Art', value: 'fantasy_art' },
  { label: 'Analog Film', value: 'analog_film' },
  { label: 'Neon Punk', value: 'neon_punk' },
  { label: 'Isometric', value: 'isometric' },
  { label: 'Low Poly', value: 'low_poly' },
  { label: 'Origami', value: 'origami' },
  { label: 'Line Art', value: 'line_art' },
  { label: 'Craft / Clay', value: 'craft_clay' },
  { label: 'Cinematic', value: 'cinematic' },
  { label: '3D Model', value: '3d_model' },
  { label: 'Pixel Art', value: 'pixel_art' },
  { label: 'Texture', value: 'texture' },
];

export default function ParamSDXLStylePresetSelect() {
  const sdxlStylePreset = useAppSelector(
    (state: RootState) => state.sdxl.sdxlStylePreset
  );

  const dispatch = useAppDispatch();

  const sdxlStylePresetChangeHandler = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      dispatch(setSDXLStylePreset(v === 'none' ? undefined : v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSearchableSelect
      label="Style Preset"
      data={sdxlPresetData}
      value={sdxlStylePreset ?? 'none'}
      placeholder="Select a preset"
      onChange={sdxlStylePresetChangeHandler}
    />
  );
}
