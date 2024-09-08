import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { MediapipeFaceProcessorConfig } from 'features/controlLayers/store/types';
import { IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { FilterComponentProps } from './types';

type Props = FilterComponentProps<MediapipeFaceProcessorConfig>;
const DEFAULTS = IMAGE_FILTERS['mediapipe_face_processor'].buildDefaults();

export const FilterMediapipeFace = memo(({ onChange, config }: Props) => {
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
        <FormLabel m={0}>{t('controlnet.maxFaces')}</FormLabel>
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
        <FormLabel m={0}>{t('controlnet.minConfidence')}</FormLabel>
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

FilterMediapipeFace.displayName = 'FilterMediapipeFace';
