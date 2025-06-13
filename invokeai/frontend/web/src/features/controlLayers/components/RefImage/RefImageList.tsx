/* eslint-disable i18next/no-literal-string */
import type { FlexProps } from '@invoke-ai/ui-library';
import { Button, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RefImage } from 'features/controlLayers/components/RefImage/RefImage';
import { RefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { useAddGlobalReferenceImage } from 'features/controlLayers/hooks/addLayerHooks';
import { selectRefImageEntityIds } from 'features/controlLayers/store/refImagesSlice';
import { memo } from 'react';
import { PiPlusBold } from 'react-icons/pi';

export const RefImageList = memo((props: FlexProps) => {
  const ids = useAppSelector(selectRefImageEntityIds);
  const addRefImage = useAddGlobalReferenceImage();
  return (
    <Flex gap={2} h={16} {...props}>
      {ids.map((id) => (
        <RefImageIdContext.Provider key={id} value={id}>
          <RefImage />
        </RefImageIdContext.Provider>
      ))}
      <Spacer />
      <Button
        size="sm"
        variant="ghost"
        h="full"
        borderWidth="2px !important"
        borderStyle="dashed !important"
        borderRadius="base"
        leftIcon={<PiPlusBold />}
        onClick={addRefImage}
        isDisabled={ids.length >= 5} // Limit to 5 reference images
      >
        Ref Image
      </Button>
    </Flex>
  );
});

RefImageList.displayName = 'RefImageList';

const AddRefImageIconButton = memo(() => {
  const addRefImage = useAddGlobalReferenceImage();
  return (
    <IconButton
      aria-label="Add reference image"
      h="full"
      variant="ghost"
      aspectRatio="1/1"
      borderWidth={2}
      borderStyle="dashed"
      borderRadius="base"
      onClick={addRefImage}
      icon={<PiPlusBold />}
    />
  );
});
AddRefImageIconButton.displayName = 'AddRefImageIconButton';

const AddRefImageButton = memo((props) => {
  const addRefImage = useAddGlobalReferenceImage();
  return (
    <Button
      size="sm"
      variant="ghost"
      h="full"
      borderWidth={2}
      borderStyle="dashed"
      borderRadius="base"
      leftIcon={<PiPlusBold />}
      onClick={addRefImage}
    >
      Ref Image
    </Button>
  );
});
AddRefImageButton.displayName = 'AddRefImageButton';
