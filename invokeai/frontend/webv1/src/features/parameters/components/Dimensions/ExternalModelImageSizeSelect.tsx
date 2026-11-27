import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  resolutionPresetSelected,
  selectAspectRatioID,
  selectImageSize,
  selectResolutionPresets,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

const makeKey = (aspectRatio: string, imageSize: string) => `${aspectRatio}|${imageSize}`;

export const ExternalModelImageSizeSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const presets = useAppSelector(selectResolutionPresets);
  const currentAspectRatio = useAppSelector(selectAspectRatioID);
  const currentImageSize = useAppSelector(selectImageSize);

  const presetMap = useMemo(() => {
    if (!presets) {
      return null;
    }
    const map = new Map<string, (typeof presets)[number]>();
    for (const preset of presets) {
      map.set(makeKey(preset.aspect_ratio, preset.image_size), preset);
    }
    return map;
  }, [presets]);

  const selectedKey = useMemo(() => {
    if (!presets || presets.length === 0) {
      return '';
    }
    if (currentImageSize && currentAspectRatio) {
      const key = makeKey(currentAspectRatio, currentImageSize);
      if (presetMap?.has(key)) {
        return key;
      }
    }
    // Fallback to first preset
    return makeKey(presets[0]!.aspect_ratio, presets[0]!.image_size);
  }, [presets, presetMap, currentAspectRatio, currentImageSize]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const preset = presetMap?.get(e.target.value);
      if (!preset) {
        return;
      }
      dispatch(
        resolutionPresetSelected({
          imageSize: preset.image_size,
          aspectRatio: preset.aspect_ratio,
          width: preset.width,
          height: preset.height,
        })
      );
    },
    [dispatch, presetMap]
  );

  if (!presets || presets.length === 0) {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('parameters.resolution')}</FormLabel>
      <Select
        size="sm"
        value={selectedKey}
        onChange={onChange}
        cursor="pointer"
        iconSize="0.75rem"
        icon={<PiCaretDownBold />}
      >
        {presets.map((preset) => {
          const key = makeKey(preset.aspect_ratio, preset.image_size);
          return (
            <option key={key} value={key}>
              {preset.label}
            </option>
          );
        })}
      </Select>
    </FormControl>
  );
});

ExternalModelImageSizeSelect.displayName = 'ExternalModelImageSizeSelect';
