import { Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import type { RequiredDWPoseImageProcessorInvocation } from 'features/controlAdapters/store/types';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.dwpose_image_processor.default as RequiredDWPoseImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredDWPoseImageProcessorInvocation;
  isEnabled: boolean;
};

const DWPoseProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { draw_body, draw_face, draw_hands } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleDrawBodyChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { draw_body: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  const handleDrawFaceChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { draw_face: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  const handleDrawHandsChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { draw_hands: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <Flex sx={{ flexDir: 'row', gap: 6 }}>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.body')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_body} isChecked={draw_body} onChange={handleDrawBodyChanged} />
        </FormControl>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.face')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_face} isChecked={draw_face} onChange={handleDrawFaceChanged} />
        </FormControl>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.hands')}</FormLabel>
          <Switch defaultChecked={DEFAULTS.draw_hands} isChecked={draw_hands} onChange={handleDrawHandsChanged} />
        </FormControl>
      </Flex>
    </ProcessorWrapper>
  );
};

export default memo(DWPoseProcessor);
