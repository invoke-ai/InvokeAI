import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/CALayer/processors/types';
import type { DWOpenposeProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { CONTROLNET_PROCESSORS } from 'features/controlLayers/util/controlAdapters';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<DWOpenposeProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['dw_openpose_image_processor'].buildDefaults();

export const DWOpenposeProcessor = memo(({ onChange, config }: Props) => {
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
    <ProcessorWrapper>
      <Flex sx={{ flexDir: 'row', gap: 6 }}>
        <FormControl w="max-content">
          <FormLabel>{t('controlnet.body')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_body} isChecked={config.draw_body} onChange={handleDrawBodyChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel>{t('controlnet.face')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_face} isChecked={config.draw_face} onChange={handleDrawFaceChanged} />
        </FormControl>
        <FormControl w="max-content">
          <FormLabel>{t('controlnet.hands')}</FormLabel>
          <Switch
            defaultChecked={DEFAULTS.draw_hands}
            isChecked={config.draw_hands}
            onChange={handleDrawHandsChanged}
          />
        </FormControl>
      </Flex>
    </ProcessorWrapper>
  );
});

DWOpenposeProcessor.displayName = 'DWOpenposeProcessor';
