import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { ProcessorComponentProps } from 'features/controlLayers/components/ControlAndIPAdapter/processors/types';
import { CONTROLNET_PROCESSORS, type MediapipeFaceProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import ProcessorWrapper from './ProcessorWrapper';

type Props = ProcessorComponentProps<MediapipeFaceProcessorConfig>;
const DEFAULTS = CONTROLNET_PROCESSORS['mediapipe_face_processor'].buildDefaults();

export const MediapipeFaceProcessor = memo(({ onChange, config }: Props) => {
  const { t } = useTranslation();

  const handleMaxFacesChanged = useCallback(
    (v: number) => {
      onChange({ ...config, max_faces: v });
    },
    [config, onChange]
  );

  const handleMinConfidenceChanged = useCallback(
    (v: number) => {
      onChange({ ...config, min_confidence: v });
    },
    [config, onChange]
  );

  return (
    <ProcessorWrapper>
      <FormControl>
        <FormLabel>{t('controlnet.maxFaces')}</FormLabel>
        <CompositeSlider
          value={config.max_faces}
          onChange={handleMaxFacesChanged}
          defaultValue={DEFAULTS.max_faces}
          min={1}
          max={20}
          marks
        />
        <CompositeNumberInput
          value={config.max_faces}
          onChange={handleMaxFacesChanged}
          defaultValue={DEFAULTS.max_faces}
          min={1}
          max={20}
        />
      </FormControl>
      <FormControl>
        <FormLabel>{t('controlnet.minConfidence')}</FormLabel>
        <CompositeSlider
          value={config.min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={DEFAULTS.min_confidence}
          min={0}
          max={1}
          step={0.01}
          marks
        />
        <CompositeNumberInput
          value={config.min_confidence}
          onChange={handleMinConfidenceChanged}
          defaultValue={DEFAULTS.min_confidence}
          min={0}
          max={1}
          step={0.01}
        />
      </FormControl>
    </ProcessorWrapper>
  );
});

MediapipeFaceProcessor.displayName = 'MediapipeFaceProcessor';
