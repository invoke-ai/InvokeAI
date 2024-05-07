import { useAppSelector } from 'app/store/storeHooks';
import { MetadataControlNets } from 'features/metadata/components/MetadataControlNets';
import { MetadataControlNetsV2 } from 'features/metadata/components/MetadataControlNetsV2';
import { MetadataIPAdapters } from 'features/metadata/components/MetadataIPAdapters';
import { MetadataIPAdaptersV2 } from 'features/metadata/components/MetadataIPAdaptersV2';
import { MetadataItem } from 'features/metadata/components/MetadataItem';
import { MetadataLoRAs } from 'features/metadata/components/MetadataLoRAs';
import { MetadataT2IAdapters } from 'features/metadata/components/MetadataT2IAdapters';
import { MetadataT2IAdaptersV2 } from 'features/metadata/components/MetadataT2IAdaptersV2';
import { handlers } from 'features/metadata/util/handlers';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

type Props = {
  metadata?: unknown;
};

const ImageMetadataActions = (props: Props) => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { metadata } = props;

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <>
      <MetadataItem metadata={metadata} handlers={handlers.generationMode} />
      <MetadataItem metadata={metadata} handlers={handlers.positivePrompt} direction="column" />
      <MetadataItem metadata={metadata} handlers={handlers.negativePrompt} direction="column" />
      <MetadataItem metadata={metadata} handlers={handlers.sdxlPositiveStylePrompt} direction="column" />
      <MetadataItem metadata={metadata} handlers={handlers.sdxlNegativeStylePrompt} direction="column" />
      <MetadataItem metadata={metadata} handlers={handlers.model} />
      <MetadataItem metadata={metadata} handlers={handlers.vae} />
      <MetadataItem metadata={metadata} handlers={handlers.width} />
      <MetadataItem metadata={metadata} handlers={handlers.height} />
      <MetadataItem metadata={metadata} handlers={handlers.seed} />
      <MetadataItem metadata={metadata} handlers={handlers.steps} />
      <MetadataItem metadata={metadata} handlers={handlers.scheduler} />
      <MetadataItem metadata={metadata} handlers={handlers.cfgScale} />
      <MetadataItem metadata={metadata} handlers={handlers.cfgRescaleMultiplier} />
      <MetadataItem metadata={metadata} handlers={handlers.initialImage} />
      <MetadataItem metadata={metadata} handlers={handlers.strength} />
      <MetadataItem metadata={metadata} handlers={handlers.hrfEnabled} />
      <MetadataItem metadata={metadata} handlers={handlers.hrfMethod} />
      <MetadataItem metadata={metadata} handlers={handlers.hrfStrength} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerCFGScale} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerModel} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerNegativeAestheticScore} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerPositiveAestheticScore} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerScheduler} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerStart} />
      <MetadataItem metadata={metadata} handlers={handlers.refinerSteps} />
      <MetadataItem metadata={metadata} handlers={handlers.layers} />
      <MetadataLoRAs metadata={metadata} />
      {activeTabName !== 'generation' && <MetadataControlNets metadata={metadata} />}
      {activeTabName !== 'generation' && <MetadataT2IAdapters metadata={metadata} />}
      {activeTabName !== 'generation' && <MetadataIPAdapters metadata={metadata} />}
      {activeTabName === 'generation' && <MetadataControlNetsV2 metadata={metadata} />}
      {activeTabName === 'generation' && <MetadataT2IAdaptersV2 metadata={metadata} />}
      {activeTabName === 'generation' && <MetadataIPAdaptersV2 metadata={metadata} />}
    </>
  );
};

export default memo(ImageMetadataActions);
