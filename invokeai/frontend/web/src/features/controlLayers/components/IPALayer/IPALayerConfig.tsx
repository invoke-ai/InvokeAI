import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ControlAdapterBeginEndStepPct } from 'features/controlLayers/components/CALayer/ControlAdapterBeginEndStepPct';
import { ControlAdapterWeight } from 'features/controlLayers/components/CALayer/ControlAdapterWeight';
import { IPAdapterImagePreview } from 'features/controlLayers/components/IPALayer/IPAdapterImagePreview';
import { IPAdapterMethod } from 'features/controlLayers/components/IPALayer/IPAdapterMethod';
import { IPAdapterModelCombobox } from 'features/controlLayers/components/IPALayer/IPALayerModelCombobox';
import {
  caOrIPALayerBeginEndStepPctChanged,
  caOrIPALayerWeightChanged,
  ipaLayerCLIPVisionModelChanged,
  ipaLayerImageChanged,
  ipaLayerMethodChanged,
  ipaLayerModelChanged,
  selectIPALayer,
} from 'features/controlLayers/store/controlLayersSlice';
import type { CLIPVisionModel, IPMethod } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback } from 'react';
import type { ImageDTO, IPAdapterModelConfig } from 'services/api/types';

type Props = {
  layerId: string;
};

export const IPALayerConfig = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const ipAdapter = useAppSelector((s) => selectIPALayer(s.controlLayers.present, layerId).ipAdapter);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(
        caOrIPALayerBeginEndStepPctChanged({
          layerId,
          beginEndStepPct,
        })
      );
    },
    [dispatch, layerId]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(caOrIPALayerWeightChanged({ layerId, weight }));
    },
    [dispatch, layerId]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethod) => {
      dispatch(ipaLayerMethodChanged({ layerId, method }));
    },
    [dispatch, layerId]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(ipaLayerModelChanged({ layerId, modelConfig }));
    },
    [dispatch, layerId]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModel) => {
      dispatch(ipaLayerCLIPVisionModelChanged({ layerId, clipVisionModel }));
    },
    [dispatch, layerId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(ipaLayerImageChanged({ layerId, imageDTO }));
    },
    [dispatch, layerId]
  );

  return (
    <Flex flexDir="column" gap={4} position="relative" w="full">
      <Flex gap={3} alignItems="center" w="full">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <IPAdapterModelCombobox
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
          <ControlAdapterWeight weight={ipAdapter.weight} onChange={onChangeWeight} />
          <ControlAdapterBeginEndStepPct
            beginEndStepPct={ipAdapter.beginEndStepPct}
            onChange={onChangeBeginEndStepPct}
          />
        </Flex>
        <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
          <IPAdapterImagePreview image={ipAdapter.image} onChangeImage={onChangeImage} layerId={layerId} />
        </Flex>
      </Flex>
    </Flex>
  );
});

IPALayerConfig.displayName = 'IPALayerConfig';
