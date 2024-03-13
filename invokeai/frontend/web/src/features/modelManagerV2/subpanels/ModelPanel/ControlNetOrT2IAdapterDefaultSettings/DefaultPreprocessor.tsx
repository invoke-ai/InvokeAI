import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ControlNetOrT2IAdapterDefaultSettingsFormData } from 'features/modelManagerV2/subpanels/ModelPanel/ControlNetOrT2IAdapterDefaultSettings/ControlNetOrT2IAdapterDefaultSettings';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

const OPTIONS = [
  { label: 'Canny', value: 'canny_image_processor' },
  { label: 'MLSD', value: 'mlsd_image_processor' },
  { label: 'Depth Anything', value: 'depth_anything_image_processor' },
  { label: 'Normal BAE', value: 'normalbae_image_processor' },
  { label: 'Pidi', value: 'pidi_image_processor' },
  { label: 'Lineart', value: 'lineart_image_processor' },
  { label: 'Lineart Anime', value: 'lineart_anime_image_processor' },
  { label: 'HED', value: 'hed_image_processor' },
  { label: 'Content Shuffle', value: 'content_shuffle_image_processor' },
  { label: 'DW OpenPose', value: 'dw_openpose_image_processor' },
  { label: 'MediaPipe Face', value: 'mediapipe_face_processor' },
  { label: 'ZoeDepth', value: 'zoe_depth_image_processor' },
  { label: 'Color Map', value: 'color_map_image_processor' },
  { label: 'None', value: 'none' },
] as const;

type DefaultSchedulerType = ControlNetOrT2IAdapterDefaultSettingsFormData['preprocessor'];

export function DefaultPreprocessor(props: UseControllerProps<ControlNetOrT2IAdapterDefaultSettingsFormData>) {
  const { t } = useTranslation();
  const { field } = useController(props);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      const updatedValue = {
        ...(field.value as FormField<string>),
        value: v.value,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => OPTIONS.find((o) => o.value === (field.value as FormField<string>).value), [field]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultSchedulerType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="controlNetProcessor">
          <FormLabel>{t('controlnet.processor')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={props.control} name="preprocessor" />
      </Flex>
      <Combobox isDisabled={isDisabled} value={value} options={OPTIONS} onChange={onChange} />
    </FormControl>
  );
}
