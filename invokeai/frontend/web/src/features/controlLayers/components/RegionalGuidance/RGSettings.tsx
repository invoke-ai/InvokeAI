import { useAppSelector } from 'app/store/storeHooks';
import { AddPromptButtons } from 'features/controlLayers/components/AddPromptButtons';
import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { selectRGOrThrow } from 'features/controlLayers/store/regionalGuidanceSlice';
import { memo } from 'react';

import { RGIPAdapters } from './RGIPAdapters';
import { RGNegativePrompt } from './RGNegativePrompt';
import { RGPositivePrompt } from './RGPositivePrompt';

type Props = {
  id: string;
};

export const RGSettings = memo(({ id }: Props) => {
  const hasPositivePrompt = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).positivePrompt !== null);
  const hasNegativePrompt = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).negativePrompt !== null);
  const hasIPAdapters = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).ipAdapters.length > 0);

  return (
    <CanvasEntitySettings>
      {!hasPositivePrompt && !hasNegativePrompt && !hasIPAdapters && <AddPromptButtons id={id} />}
      {hasPositivePrompt && <RGPositivePrompt id={id} />}
      {hasNegativePrompt && <RGNegativePrompt id={id} />}
      {hasIPAdapters && <RGIPAdapters id={id} />}
    </CanvasEntitySettings>
  );
});

RGSettings.displayName = 'RGSettings';
