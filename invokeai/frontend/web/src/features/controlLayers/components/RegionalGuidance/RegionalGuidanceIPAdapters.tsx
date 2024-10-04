import { Divider } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { RegionalGuidanceIPAdapterSettings } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceIPAdapterSettings';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { Fragment, memo, useMemo } from 'react';

export const RegionalGuidanceIPAdapters = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');

  const selectIPAdapterIds = useMemo(
    () =>
      createMemoizedSelector(selectCanvasSlice, (canvas) => {
        const ipAdapterIds = selectEntityOrThrow(canvas, entityIdentifier).referenceImages.map(({ id }) => id);
        if (ipAdapterIds.length === 0) {
          return EMPTY_ARRAY;
        }
        return ipAdapterIds;
      }),
    [entityIdentifier]
  );

  const ipAdapterIds = useAppSelector(selectIPAdapterIds);

  if (ipAdapterIds.length === 0) {
    return null;
  }

  return (
    <>
      {ipAdapterIds.map((ipAdapterId, index) => (
        <Fragment key={ipAdapterId}>
          {index > 0 && <Divider />}
          <RegionalGuidanceIPAdapterSettings referenceImageId={ipAdapterId} />
        </Fragment>
      ))}
    </>
  );
});

RegionalGuidanceIPAdapters.displayName = 'RegionalGuidanceLayerIPAdapterList';
