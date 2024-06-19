import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAMethod } from 'features/controlLayers/components/IPAdapter/IPAMethod';
import {
  ipaBeginEndStepPctChanged,
  ipaCLIPVisionModelChanged,
  ipaImageChanged,
  ipaMethodChanged,
  ipaModelChanged,
  ipaWeightChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import { selectIPAOrThrow } from 'features/controlLayers/store/ipAdaptersReducers';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/store/types';
import type { IPAImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import type { ImageDTO, IPAdapterModelConfig, IPALayerImagePostUploadAction } from 'services/api/types';

import { IPAImagePreview } from './IPAImagePreview';
import { IPAModelCombobox } from './IPAModelCombobox';

type Props = {
  id: string;
};

export const IPASettings = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const ipAdapter = useAppSelector((s) => selectIPAOrThrow(s.canvasV2, id));

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(ipaBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(ipaWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(ipaMethodChanged({ id, method }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(ipaModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(ipaCLIPVisionModelChanged({ id, clipVisionModel }));
    },
    [dispatch, id]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(ipaImageChanged({ id, imageDTO }));
    },
    [dispatch, id]
  );

  const droppableData = useMemo<IPAImageDropData>(() => ({ actionType: 'SET_IPA_IMAGE', context: { id }, id }), [id]);
  const postUploadAction = useMemo<IPALayerImagePostUploadAction>(() => ({ type: 'SET_IPA_IMAGE', id }), [id]);

  return (
    <CanvasEntitySettings>
      <Flex flexDir="column" gap={4} position="relative" w="full">
        <Flex gap={3} alignItems="center" w="full">
          <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
            <IPAModelCombobox
              modelKey={ipAdapter.model?.key ?? null}
              onChangeModel={onChangeModel}
              clipVisionModel={ipAdapter.clipVisionModel}
              onChangeCLIPVisionModel={onChangeCLIPVisionModel}
            />
          </Box>
        </Flex>
        <Flex gap={4} w="full" alignItems="center">
          <Flex flexDir="column" gap={3} w="full">
            <IPAMethod method={ipAdapter.method} onChange={onChangeIPMethod} />
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
            <IPAImagePreview
              image={ipAdapter.imageObject?.image ?? null}
              onChangeImage={onChangeImage}
              ipAdapterId={ipAdapter.id}
              droppableData={droppableData}
              postUploadAction={postUploadAction}
            />
          </Flex>
        </Flex>
      </Flex>
    </CanvasEntitySettings>
  );
});

IPASettings.displayName = 'IPASettings';
