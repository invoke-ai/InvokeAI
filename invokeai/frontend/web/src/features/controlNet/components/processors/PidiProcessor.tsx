import { useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import IAISwitch from 'common/components/IAISwitch';
import { CONTROLNET_PROCESSORS } from 'features/controlNet/store/constants';
import { RequiredPidiImageProcessorInvocation } from 'features/controlNet/store/types';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { ChangeEvent, memo, useCallback } from 'react';
import { useProcessorNodeChanged } from '../hooks/useProcessorNodeChanged';
import ProcessorWrapper from './common/ProcessorWrapper';

const DEFAULTS = CONTROLNET_PROCESSORS.pidi_image_processor
  .default as RequiredPidiImageProcessorInvocation;

type Props = {
  controlNetId: string;
  processorNode: RequiredPidiImageProcessorInvocation;
  isEnabled: boolean;
};

const PidiProcessor = (props: Props) => {
  const { controlNetId, processorNode, isEnabled } = props;
  const { image_resolution, detect_resolution, scribble, safe } = processorNode;
  const processorChanged = useProcessorNodeChanged();
  const isBusy = useAppSelector(selectIsBusy);

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

  const handleScribbleChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { scribble: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  const handleSafeChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      processorChanged(controlNetId, { safe: e.target.checked });
    },
    [controlNetId, processorChanged]
  );

  return (
    <ProcessorWrapper>
      <IAISlider
        label="Detect Resolution"
        value={detect_resolution}
        onChange={handleDetectResolutionChanged}
        handleReset={handleDetectResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={isBusy || !isEnabled}
      />
      <IAISlider
        label="Image Resolution"
        value={image_resolution}
        onChange={handleImageResolutionChanged}
        handleReset={handleImageResolutionReset}
        withReset
        min={0}
        max={4096}
        withInput
        withSliderMarks
        isDisabled={isBusy || !isEnabled}
      />
      <IAISwitch
        label="Scribble"
        isChecked={scribble}
        onChange={handleScribbleChanged}
      />
      <IAISwitch
        label="Safe"
        isChecked={safe}
        onChange={handleSafeChanged}
        isDisabled={isBusy || !isEnabled}
      />
    </ProcessorWrapper>
  );
};

export default memo(PidiProcessor);
