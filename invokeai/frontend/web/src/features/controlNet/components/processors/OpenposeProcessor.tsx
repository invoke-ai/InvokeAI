import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredOpenposeImageProcessorInvocation } from 'features/controlNet/store/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.openpose_image_processor
  .default as RequiredOpenposeImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredOpenposeImageProcessorInvocation;
  isEnabled: boolean;
};

const OpenposeProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, detect_resolution, hand_and_face } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const { t } = useTranslation();

  const handleDetectResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { detect_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleImageResolutionChanged = useCallback(
    (v: number) => {
      processorChanged(controlNetId, { image_resolution: v });
    },
    [controlNetId, processorChanged]
  );

  const handleDetectResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      detect_resolution: DEFAULTS.detect_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleImageResolutionReset = useCallback(() => {
    processorChanged(controlNetId, {
      image_resolution: DEFAULTS.image_resolution,
    });
  }, [controlNetId, processorChanged]);

  const handleHandAndFaceChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { hand_and_face: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <IAISlider
        label={t('controlnet.detectResolution')}
        value={detect_resolution}
        onChange={handleDetectResolutionChanged}
        handleReset={handleDetectResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISlider
        label={t('controlnet.imageResolution')}
        value={image_resolution}
        onChange={handleImageResolutionChanged}
        handleReset={handleImageResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={!isEnabled}
      />
      <IAISwitch
        label={t('controlnet.handAndFace')}
        isChecked={hand_and_face}
        onChange={handleHandAndFaceChanged}
        isDisabled={!isEnabled}
      />
    </ProcessorWrapper>
  );
};

export default memo(OpenposeProcessor);
