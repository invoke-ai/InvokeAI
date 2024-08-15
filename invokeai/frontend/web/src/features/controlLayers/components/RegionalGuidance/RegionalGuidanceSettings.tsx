import { useAppSelector } from 'app/store/storeHooks';
import { AddPromptButtons } from 'features/controlLayers/components/AddPromptButtons';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo } from 'react';

import { RegionalGuidanceIPAdapters } from './RegionalGuidanceIPAdapters';
import { RegionalGuidanceNegativePrompt } from './RegionalGuidanceNegativePrompt';
import { RegionalGuidancePositivePrompt } from './RegionalGuidancePositivePrompt';

export const RegionalGuidanceSettings = memo(() => {
  const { id } = useEntityIdentifierContext();
  const hasPositivePrompt = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).positivePrompt !== null);
  const hasNegativePrompt = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).negativePrompt !== null);
  const hasIPAdapters = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).ipAdapters.length > 0);

  return (
    <CanvasEntitySettingsWrapper>
      {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && <AddPromptButtons id={id} />}
      {hasPositivePrompt && <RegionalGuidancePositivePrompt id={id} />}
      {hasNegativePrompt && <RegionalGuidanceNegativePrompt id={id} />}
      {hasIPAdapters && <RegionalGuidanceIPAdapters id={id} />}
    </CanvasEntitySettingsWrapper>
  );
});

RegionalGuidanceSettings.displayName = 'RegionalGuidanceSettings';
