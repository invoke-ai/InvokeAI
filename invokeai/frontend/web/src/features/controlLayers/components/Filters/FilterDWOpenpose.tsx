import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { DWOpenposeProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<DWOpenposeProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['dw_openpose_image_processor'].buildDefaults();

export const FilterDWOpenpose = memo(({ onChange, config }: Props) => {
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
          <FormLabel m={0}>{t('controlnet.body')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_body} isChecked={config.draw_body} onChange={handleDrawBodyChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel m={0}>{t('controlnet.face')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_face} isChecked={config.draw_face} onChange={handleDrawFaceChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel m={0}>{t('controlnet.hands')}</FormLabel>
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

FilterDWOpenpose.displayName = 'FilterDWOpenpose';
