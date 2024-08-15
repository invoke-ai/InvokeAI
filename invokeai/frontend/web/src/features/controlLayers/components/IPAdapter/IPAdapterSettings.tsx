import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterMethod } from 'features/controlLayers/components/IPAdapter/IPAdapterMethod';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
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

import { IPAdapterImagePreview } from './IPAdapterImagePreview';
import { IPAdapterModel } from './IPAdapterModel';

export const IPAdapterSettings = memo(() => {
  const dispatch = useAppDispatch();
  const { id } = useEntityIdentifierContext();
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
    <CanvasEntitySettingsWrapper>
      <Flex flexDir="column" gap={4} position="relative" w="full">
        <Flex gap={3} alignItems="center" w="full">
          <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
            <IPAdapterModel
              modelKey={ipAdapter.model?.key ?? null}
              onChangeModel={onChangeModel}
              clipVisionModel={ipAdapter.clipVisionModel}
              onChangeCLIPVisionModel={onChangeCLIPVisionModel}
            />
          </Box>
        </Flex>
        <Flex gap={4} w="full" alignItems="center">
          <Flex flexDir="column" gap={3} w="full">
            <IPAdapterMethod method={ipAdapter.method} onChange={onChangeIPMethod} />
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
            <IPAdapterImagePreview
              image={ipAdapter.imageObject?.image ?? null}
              onChangeImage={onChangeImage}
              ipAdapterId={ipAdapter.id}
              droppableData={droppableData}
              postUploadAction={postUploadAction}
            />
          </Flex>
        </Flex>
      </Flex>
    </CanvasEntitySettingsWrapper>
  );
});

IPAdapterSettings.displayName = 'IPAdapterSettings';
