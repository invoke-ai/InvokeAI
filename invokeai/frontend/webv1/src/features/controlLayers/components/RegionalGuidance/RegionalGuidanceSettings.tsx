import { Divider } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { RegionalGuidanceAddPromptsIPAdapterButtons } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceAddPromptsIPAdapterButtons';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

import { RegionalGuidanceIPAdapters } from './RegionalGuidanceIPAdapters';
import { RegionalGuidanceNegativePrompt } from './RegionalGuidanceNegativePrompt';
import { RegionalGuidancePositivePrompt } from './RegionalGuidancePositivePrompt';

const buildSelectFlags = (entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>) =>
  createMemoizedSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'RegionalGuidanceSettings');
    return {
      hasPositivePrompt: entity.positivePrompt !== null,
      hasNegativePrompt: entity.negativePrompt !== null,
      hasIPAdapters: entity.referenceImages.length > 0,
    };
  });
export const RegionalGuidanceSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const selectFlags = useMemo(() => buildSelectFlags(entityIdentifier), [entityIdentifier]);
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
