import { Divider } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { RegionalGuidanceAddPromptsIPAdapterButtons } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceAddPromptsIPAdapterButtons';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

import { RegionalGuidanceIPAdapters } from './RegionalGuidanceIPAdapters';
import { RegionalGuidanceNegativePrompt } from './RegionalGuidanceNegativePrompt';
import { RegionalGuidancePositivePrompt } from './RegionalGuidancePositivePrompt';

export const RegionalGuidanceSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const hasPositivePrompt = useAppSelector(
    (s) => selectEntityOrThrow(s.canvasV2, entityIdentifier).positivePrompt !== null
  );
  const hasNegativePrompt = useAppSelector(
    (s) => selectEntityOrThrow(s.canvasV2, entityIdentifier).negativePrompt !== null
  );
  const hasIPAdapters = useAppSelector((s) => selectEntityOrThrow(s.canvasV2, entityIdentifier).ipAdapters.length > 0);

  return (
    <CanvasEntitySettingsWrapper>
      {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && <RegionalGuidanceAddPromptsIPAdapterButtons />}
      {hasPositivePrompt && (
        <>
          <RegionalGuidancePositivePrompt />
          {(hasNegativePrompt || hasIPAdapters) && <Divider />}
        </>
      )}
      {hasNegativePrompt && (
        <>
          <RegionalGuidanceNegativePrompt />
          {hasIPAdapters && <Divider />}
        </>
      )}
      {hasIPAdapters && <RegionalGuidanceIPAdapters />}
    </CanvasEntitySettingsWrapper>
  );
});

RegionalGuidanceSettings.displayName = 'RegionalGuidanceSettings';
