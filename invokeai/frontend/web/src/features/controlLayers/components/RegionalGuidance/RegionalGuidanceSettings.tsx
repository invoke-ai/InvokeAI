import { Divider } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { RegionalGuidanceAddPromptsIPAdapterButtons } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceAddPromptsIPAdapterButtons';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useMemo } from 'react';

import { RegionalGuidanceIPAdapters } from './RegionalGuidanceIPAdapters';
import { RegionalGuidanceNegativePrompt } from './RegionalGuidanceNegativePrompt';
import { RegionalGuidancePositivePrompt } from './RegionalGuidancePositivePrompt';

export const RegionalGuidanceSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const selectFlags = useMemo(
    () =>
      createMemoizedSelector(selectCanvasSlice, (canvas) => {
        const entity = selectEntityOrThrow(canvas, entityIdentifier);
        return {
          hasPositivePrompt: entity.positivePrompt !== null,
          hasNegativePrompt: entity.negativePrompt !== null,
          hasIPAdapters: entity.referenceImages.length > 0,
        };
      }),
    [entityIdentifier]
  );
  const flags = useAppSelector(selectFlags);

  return (
    <CanvasEntitySettingsWrapper>
      {!flags.hasPositivePrompt && !flags.hasNegativePrompt && !flags.hasIPAdapters && (
        <RegionalGuidanceAddPromptsIPAdapterButtons />
      )}
      {flags.hasPositivePrompt && (
        <>
          <RegionalGuidancePositivePrompt />
          {!flags.hasNegativePrompt && flags.hasIPAdapters && <Divider />}
        </>
      )}
      {flags.hasNegativePrompt && (
        <>
          <RegionalGuidanceNegativePrompt />
          {flags.hasIPAdapters && <Divider />}
        </>
      )}
      {flags.hasIPAdapters && <RegionalGuidanceIPAdapters />}
    </CanvasEntitySettingsWrapper>
  );
});

RegionalGuidanceSettings.displayName = 'RegionalGuidanceSettings';
