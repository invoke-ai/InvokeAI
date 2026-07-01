import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { type DWOpenposeDetectionFilterConfig, IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<DWOpenposeDetectionFilterConfig>;
const DEFAULTS = IMAGE_FILTERS['dw_openpose_detection'].buildDefaults();

export const FilterDWOpenposeDetection = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleDrawBodyChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, draw_body: e.target.checked });
    },
    [config, onChange]
  );

  const handleDrawFaceChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, draw_face: e.target.checked });
    },
    [config, onChange]
  );

  const handleDrawHandsChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...config, draw_hands: e.target.checked });
    },
    [config, onChange]
  );

  return (
    <>
      <Flex sx={{ flexDir: 'row', gap: 6 }}>
        <FormControl w="max-content">
          <FormLabel m={0}>{t('controlLayers.filter.dw_openpose_detection.draw_body')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_body} isChecked={config.draw_body} onChange={handleDrawBodyChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel m={0}>{t('controlLayers.filter.dw_openpose_detection.draw_face')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_face} isChecked={config.draw_face} onChange={handleDrawFaceChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel m={0}>{t('controlLayers.filter.dw_openpose_detection.draw_hands')}</FormLabel>
          <Switch
            defaultChecked={DEFAULTS.draw_hands}
            isChecked={config.draw_hands}
            onChange={handleDrawHandsChanged}
          />
        </FormControl>
      </Flex>
    </>
  );
});

FilterDWOpenposeDetection.displayName = 'FilterDWOpenposeDetection';
