import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useProcessorNodeChanged } from 'features/controlAdapters/components/hooks/useProcessorNodeChanged';
import { useGetDefaultForControlnetProcessor } from 'features/controlAdapters/hooks/useGetDefaultForControlnetProcessor';
import type { RequiredDWOpenposeImageProcessorInvocation } from 'features/controlAdapters/store/types';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './common/ProcessorWrapper';

type Props = {
  controlNetId: string;
  processorNode: RequiredDWOpenposeImageProcessorInvocation;
  isEnabled: boolean;
};

const DWOpenposeProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, draw_body, draw_face, draw_hands } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const defaults = useGetDefaultForControlnetProcessor(
    'dw_openpose_image_processor'
  ) as RequiredDWOpenposeImageProcessorInvocation;

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

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <Flex sx={{ flexDir: 'row', gap: 6 }}>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.body')}</FormLabel>
          <Switch defaultChecked={defaults.draw_body} isChecked={draw_body} onChange={handleDrawBodyChanged} />
        </FormControl>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.face')}</FormLabel>
          <Switch defaultChecked={defaults.draw_face} isChecked={draw_face} onChange={handleDrawFaceChanged} />
        </FormControl>
        <FormControl isDisabled={!isEnabled} w="max-content">
          <FormLabel>{t('controlnet.hands')}</FormLabel>
          <Switch defaultChecked={defaults.draw_hands} isChecked={draw_hands} onChange={handleDrawHandsChanged} />
        </FormControl>
      </Flex>
      <FormControl isDisabled={!isEnabled}>
        <FormLabel>{t('controlnet.imageResolution')}</FormLabel>
        <CompositeSlider
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
          marks
        />
        <CompositeNumberInput
          value={image_resolution}
          onChange={handleImageResolutionChanged}
          defaultValue={defaults.image_resolution}
          min={0}
          max={4096}
        />
      </FormControl>
    </ProcessorWrapper>
  );
};

export default memo(DWOpenposeProcessor);
