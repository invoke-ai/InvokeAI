import { Box, Flex } from '@invoke-ai/ui-library';
import { BeginEndStepPct } from 'features/controlLayers/components/ControlAndIPAdapter/ControlAdapterBeginEndStepPct';
import { ControlAdapterWeight } from 'features/controlLayers/components/ControlAndIPAdapter/ControlAdapterWeight';
import { IPAdapterImagePreview } from 'features/controlLayers/components/ControlAndIPAdapter/IPAdapterImagePreview';
import { IPAdapterMethod } from 'features/controlLayers/components/ControlAndIPAdapter/IPAdapterMethod';
import { IPAdapterModelSelect } from 'features/controlLayers/components/ControlAndIPAdapter/IPAdapterModelSelect';
import type { CLIPVisionModelV2, IPAdapterConfigV2, IPMethodV2 } from 'features/controlLayers/util/controlAdapters';
import type { TypesafeDroppableData } from 'features/dnd/types';
import { memo } from 'react';
import type { ImageDTO, IPAdapterModelConfig, PostUploadAction } from 'services/api/types';

type Props = {
  ipAdapter: IPAdapterConfigV2;
  onChangeBeginEndStepPct: (beginEndStepPct: [number, number]) => void;
  onChangeWeight: (weight: number) => void;
  onChangeIPMethod: (method: IPMethodV2) => void;
  onChangeModel: (modelConfig: IPAdapterModelConfig) => void;
  onChangeCLIPVisionModel: (clipVisionModel: CLIPVisionModelV2) => void;
  onChangeImage: (imageDTO: ImageDTO | null) => void;
  droppableData: TypesafeDroppableData;
  postUploadAction: PostUploadAction;
};

export const IPAdapter = memo(
  ({
    ipAdapter,
    onChangeBeginEndStepPct,
    onChangeWeight,
    onChangeIPMethod,
    onChangeModel,
    onChangeCLIPVisionModel,
    onChangeImage,
    droppableData,
    postUploadAction,
  }: Props) => {
    return (
      <Flex flexDir="column" gap={4} position="relative" w="full">
        <Flex gap={3} alignItems="center" w="full">
          <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
            <IPAdapterModelSelect
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
            <BeginEndStepPct
              beginEndStepPct={ipAdapter.beginEndStepPct}
              onChange={onChangeBeginEndStepPct}
            />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
            <IPAdapterImagePreview
              image={ipAdapter.image}
              onChangeImage={onChangeImage}
              ipAdapterId={ipAdapter.id}
              droppableData={droppableData}
              postUploadAction={postUploadAction}
            />
          </Flex>
        </Flex>
      </Flex>
    );
  }
);

IPAdapter.displayName = 'IPAdapter';
