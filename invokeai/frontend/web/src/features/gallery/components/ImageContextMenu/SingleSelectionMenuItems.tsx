import { MenuDivider } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { ImageMenuItemChangeBoard } from 'features/gallery/components/ImageContextMenu/ImageMenuItemChangeBoard';
import { ImageMenuItemCopy } from 'features/gallery/components/ImageContextMenu/ImageMenuItemCopy';
import { ImageMenuItemDelete } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDelete';
import { ImageMenuItemDownload } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDownload';
import { ImageMenuItemLoadWorkflow } from 'features/gallery/components/ImageContextMenu/ImageMenuItemLoadWorkflow';
import { ImageMenuItemLocateInGalery } from 'features/gallery/components/ImageContextMenu/ImageMenuItemLocateInGalery';
import { ImageMenuItemMetadataRecallActionsCanvasGenerateTabs } from 'features/gallery/components/ImageContextMenu/ImageMenuItemMetadataRecallActionsCanvasGenerateTabs';
import { ImageMenuItemNewCanvasFromImageSubMenu } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewCanvasFromImageSubMenu';
import { ImageMenuItemNewLayerFromImageSubMenu } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewLayerFromImageSubMenu';
import { ImageMenuItemOpenInNewTab } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInNewTab';
import { ImageMenuItemOpenInViewer } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInViewer';
import { ImageMenuItemSelectForCompare } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSelectForCompare';
import { ImageMenuItemSendToUpscale } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSendToUpscale';
import { ImageMenuItemStarUnstar } from 'features/gallery/components/ImageContextMenu/ImageMenuItemStarUnstar';
import { ImageMenuItemUseAsRefImage } from 'features/gallery/components/ImageContextMenu/ImageMenuItemUseAsRefImage';
import { ImageMenuItemUseForPromptGeneration } from 'features/gallery/components/ImageContextMenu/ImageMenuItemUseForPromptGeneration';
import { ImageDTOContextProvider } from 'features/gallery/contexts/ImageDTOContext';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import type { ImageDTO } from 'services/api/types';

import { ImageMenuItemMetadataRecallActionsUpscaleTab } from './ImageMenuItemMetadataRecallActionsUpscaleTab';
import { ImageMenuItemUseAsPromptTemplate } from './ImageMenuItemUseAsPromptTemplate';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = ({ imageDTO }: SingleSelectionMenuItemsProps) => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <ImageDTOContextProvider value={imageDTO}>
      <IconMenuItemGroup>
        <ImageMenuItemOpenInNewTab />
        <ImageMenuItemCopy />
        <ImageMenuItemDownload />
        <ImageMenuItemOpenInViewer />
        <ImageMenuItemSelectForCompare />
        <ImageMenuItemDelete />
      </IconMenuItemGroup>
      <MenuDivider />
      <ImageMenuItemLoadWorkflow />
      {(tab === 'canvas' || tab === 'generate') && <ImageMenuItemMetadataRecallActionsCanvasGenerateTabs />}
      {tab === 'upscaling' && <ImageMenuItemMetadataRecallActionsUpscaleTab />}
      <MenuDivider />
      <ImageMenuItemSendToUpscale />
      <ImageMenuItemUseForPromptGeneration />
      {(tab === 'canvas' || tab === 'generate') && <ImageMenuItemUseAsRefImage />}
      <ImageMenuItemUseAsPromptTemplate />
      <ImageMenuItemNewCanvasFromImageSubMenu />
      {tab === 'canvas' && <ImageMenuItemNewLayerFromImageSubMenu />}
      <MenuDivider />
      <ImageMenuItemChangeBoard />
      <ImageMenuItemStarUnstar />
      {(tab === 'canvas' || tab === 'generate' || tab === 'workflows' || tab === 'upscaling') &&
        !imageDTO.is_intermediate && (
          // Only render this button on tabs with a gallery.
          <ImageMenuItemLocateInGalery />
        )}
    </ImageDTOContextProvider>
  );
};

export default memo(SingleSelectionMenuItems);
