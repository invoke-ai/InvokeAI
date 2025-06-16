import type { FlexProps } from '@invoke-ai/ui-library';
import { Button, Flex } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { RefImage } from 'features/controlLayers/components/RefImage/RefImage';
import { RefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { getDefaultRefImageConfig } from 'features/controlLayers/hooks/addLayerHooks';
import { refImageAdded, selectRefImageEntityIds } from 'features/controlLayers/store/refImagesSlice';
import { imageDTOToImageWithDims } from 'features/controlLayers/store/util';
import { addGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { memo, useMemo } from 'react';
import { PiUploadBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const RefImageList = memo((props: FlexProps) => {
  const ids = useAppSelector(selectRefImageEntityIds);
  return (
    <Flex gap={2} h={16} {...props}>
      {ids.map((id) => (
        <RefImageIdContext.Provider key={id} value={id}>
          <RefImage />
        </RefImageIdContext.Provider>
      ))}
      {ids.length < 5 && <AddRefImageDropTargetAndButton />}
      {ids.length >= 5 && <MaxRefImages />}
    </Flex>
  );
});

RefImageList.displayName = 'RefImageList';

const dndTargetData = addGlobalReferenceImageDndTarget.getData();

const MaxRefImages = memo(() => {
  return (
    <Button
      position="relative"
      size="sm"
      variant="ghost"
      h="full"
      w="full"
      borderWidth="2px !important"
      borderStyle="dashed !important"
      borderRadius="base"
      isDisabled
    >
      Max Ref Images
    </Button>
  );
});
MaxRefImages.displayName = 'MaxRefImages';

const AddRefImageDropTargetAndButton = memo(() => {
  const { dispatch, getState } = useAppStore();

  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          const config = getDefaultRefImageConfig(getState);
          config.image = imageDTOToImageWithDims(imageDTO);
          dispatch(refImageAdded({ overrides: { config } }));
        },
        allowMultiple: false,
      }) as const,
    [dispatch, getState]
  );

  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <>
      <Button
        position="relative"
        size="sm"
        variant="ghost"
        h="full"
        w="full"
        borderWidth="2px !important"
        borderStyle="dashed !important"
        borderRadius="base"
        leftIcon={<PiUploadBold />}
        {...uploadApi.getUploadButtonProps()}
      >
        Reference Image
        <input {...uploadApi.getUploadInputProps()} />
        <DndDropTarget label="Drop" dndTarget={addGlobalReferenceImageDndTarget} dndTargetData={dndTargetData} />
      </Button>
    </>
  );
});
AddRefImageDropTargetAndButton.displayName = 'AddRefImageDropTargetAndButton';
