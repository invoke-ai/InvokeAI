import { MetadataControlNets } from 'features/metadata/components/MetadataControlNets';
import { MetadataIPAdapters } from 'features/metadata/components/MetadataIPAdapters';
import { MetadataItem } from 'features/metadata/components/MetadataItem';
import { MetadataLoRAs } from 'features/metadata/components/MetadataLoRAs';
import { MetadataT2IAdapters } from 'features/metadata/components/MetadataT2IAdapters';
import { handlers } from 'features/metadata/util/handlers';
import { memo } from 'react';

type Props = {
  metadata?: unknown;
};

const ImageMetadataActions = (props: Props) => {
  const { metadata } = props;

  if (!metadata || Object.keys(metadata).length === 0) {
    return null;
  }

  return (
    <>
      <MetadataItem metadata={metadata} handlers={handlers.createdBy} />
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
      <MetadataLoRAs metadata={metadata} />
      <MetadataControlNets metadata={metadata} />
      <MetadataT2IAdapters metadata={metadata} />
      <MetadataIPAdapters metadata={metadata} />
    </>
  );
};

export default memo(ImageMetadataActions);
