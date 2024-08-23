import { Divider } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { RegionalGuidanceAddPromptsIPAdapterButtons } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceAddPromptsIPAdapterButtons';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectRegionalGuidanceEntityOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo } from 'react';

import { RegionalGuidanceIPAdapters } from './RegionalGuidanceIPAdapters';
import { RegionalGuidanceNegativePrompt } from './RegionalGuidanceNegativePrompt';
import { RegionalGuidancePositivePrompt } from './RegionalGuidancePositivePrompt';

export const RegionalGuidanceSettings = memo(() => {
  const { id } = useEntityIdentifierContext();
  const hasPositivePrompt = useAppSelector(
    (s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, id).positivePrompt !== null
  );
  const hasNegativePrompt = useAppSelector(
    (s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, id).negativePrompt !== null
  );
  const hasIPAdapters = useAppSelector(
    (s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, id).ipAdapters.length > 0
  );

  return (
    <CanvasEntitySettingsWrapper>
      {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && (
        <RegionalGuidanceAddPromptsIPAdapterButtons id={id} />
      )}
      {hasPositivePrompt && (
        <>
          <RegionalGuidancePositivePrompt id={id} />
          {(hasNegativePrompt || hasIPAdapters) && <Divider />}
        </>
      )}
      {hasNegativePrompt && (
        <>
          <RegionalGuidanceNegativePrompt id={id} />
          {hasIPAdapters && <Divider />}
        </>
      )}
      {hasIPAdapters && <RegionalGuidanceIPAdapters id={id} />}
    </CanvasEntitySettingsWrapper>
  );
});

RegionalGuidanceSettings.displayName = 'RegionalGuidanceSettings';
