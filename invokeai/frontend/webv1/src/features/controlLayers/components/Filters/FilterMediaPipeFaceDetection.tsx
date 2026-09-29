import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { MediaPipeFaceDetectionFilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<MediaPipeFaceDetectionFilterConfig>;
const DEFAULTS = IMAGE_FILTERS.mediapipe_face_detection.buildDefaults();

export const FilterMediaPipeFaceDetection = memo(({ onChange, config }: Props) => {
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
    <>
      <FormControl>
        <FormLabel m={0}>{t('controlLayers.filter.mediapipe_face_detection.max_faces')}</FormLabel>
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
        <FormLabel m={0}>{t('controlLayers.filter.mediapipe_face_detection.min_confidence')}</FormLabel>
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
    </>
  );
});

FilterMediaPipeFaceDetection.displayName = 'FilterMediaPipeFaceDetection';
