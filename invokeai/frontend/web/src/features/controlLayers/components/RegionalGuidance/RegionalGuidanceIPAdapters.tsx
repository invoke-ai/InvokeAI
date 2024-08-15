import { Divider } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { RegionalGuidanceIPAdapterSettings } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceIPAdapterSettings';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { selectRegionalGuidanceEntityOrThrow } from 'features/controlLayers/store/regionsReducers';
import { Fragment, memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const RegionalGuidanceIPAdapters = memo(({ id }: Props) => {
  const selectIPAdapterIds = useMemo(
    () =>
      createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => {
        const ipAdapterIds = selectRegionalGuidanceEntityOrThrow(canvasV2, id).ipAdapters.map(({ id }) => id);
        if (ipAdapterIds.length === 0) {
          return EMPTY_ARRAY;
        }
        return ipAdapterIds;
      }),
    [id]
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
          <RegionalGuidanceIPAdapterSettings id={id} ipAdapterId={ipAdapterId} ipAdapterNumber={index + 1} />
        </Fragment>
      ))}
    </>
  );
});

RegionalGuidanceIPAdapters.displayName = 'RegionalGuidanceLayerIPAdapterList';
