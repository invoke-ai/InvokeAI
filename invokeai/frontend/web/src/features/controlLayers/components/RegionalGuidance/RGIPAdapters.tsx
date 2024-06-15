import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RGIPAdapterSettings } from 'features/controlLayers/components/RegionalGuidance/RGIPAdapterSettings';
import { selectRGOrThrow } from 'features/controlLayers/store/regionalGuidanceSlice';
import { memo } from 'react';

type Props = {
  id: string;
};

export const RGIPAdapters = memo(({ id }: Props) => {
  const ipAdapterIds = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).ipAdapters.map(({ id }) => id));

  if (ipAdapterIds.length === 0) {
    return null;
  }

  return (
    <>
      {ipAdapterIds.map((id, index) => (
        <Flex flexDir="column" key={id}>
          {index > 0 && (
            <Flex pb={3}>
              <Divider />
            </Flex>
          )}
          <RGIPAdapterSettings id={id} ipAdapterId={id} ipAdapterNumber={index + 1} />
        </Flex>
      ))}
    </>
  );
});

RGIPAdapters.displayName = 'RGLayerIPAdapterList';
