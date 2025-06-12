/* eslint-disable i18next/no-literal-string */
import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { RefImagePreview } from 'features/controlLayers/components/IPAdapter/IPAdapterPreview';
import { RefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectRefImageEntityIds } from 'features/controlLayers/store/refImagesSlice';
import { memo } from 'react';

const sx: SystemStyleObject = {
  opacity: 0.3,
  _hover: {
    opacity: 1,
  },
  transitionProperty: 'opacity',
  transitionDuration: '0.2s',
};

export const RefImageList = memo((props: FlexProps) => {
  const ids = useAppSelector(selectRefImageEntityIds);

  if (ids.length === 0) {
    return null;
  }

  return (
    <Flex gap={2} {...props}>
      {ids.map((id) => (
        <RefImageIdContext.Provider key={id} value={id}>
          <RefImagePreview />
        </RefImageIdContext.Provider>
      ))}
    </Flex>
  );
});

RefImageList.displayName = 'RefImageList';
