import { MenuDivider } from '@invoke-ai/ui-library';
import { ImageMenuItemChangeBoard } from 'features/gallery/components/ImageContextMenu/ImageMenuItemChangeBoard';
import { ImageMenuItemCopy } from 'features/gallery/components/ImageContextMenu/ImageMenuItemCopy';
import { ImageMenuItemDelete } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDelete';
import { ImageMenuItemDownload } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDownload';
import { ImageMenuItemLoadWorkflow } from 'features/gallery/components/ImageContextMenu/ImageMenuItemLoadWorkflow';
import { ImageMenuItemMetadataRecallActions } from 'features/gallery/components/ImageContextMenu/ImageMenuItemMetadataRecallActions';
import { ImageMenuItemNewCanvasFromImage } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewCanvasFromImage';
import { ImageMenuItemNewLayerFromImage } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewLayerFromImage';
import { ImageMenuItemOpenInNewTab } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInNewTab';
import { ImageMenuItemOpenInViewer } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInViewer';
import { ImageMenuItemSelectForCompare } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSelectForCompare';
import { ImageMenuItemSendToUpscale } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSendToUpscale';
import { ImageMenuItemStarUnstar } from 'features/gallery/components/ImageContextMenu/ImageMenuItemStarUnstar';
import { ImageDTOContextProvider } from 'features/gallery/contexts/ImageDTOContext';
import { memo } from 'react';
import type { ImageDTO } from 'services/api/types';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = ({ imageDTO }: SingleSelectionMenuItemsProps) => {
  return (
    <ImageDTOContextProvider value={imageDTO}>
      <ImageMenuItemOpenInNewTab />
      <ImageMenuItemCopy />
      <ImageMenuItemDownload />
      <ImageMenuItemOpenInViewer />
      <ImageMenuItemSelectForCompare />
      <MenuDivider />
      <ImageMenuItemLoadWorkflow />
      <ImageMenuItemMetadataRecallActions />
      <MenuDivider />
      <ImageMenuItemSendToUpscale />
      <ImageMenuItemNewLayerFromImage />
      <ImageMenuItemNewCanvasFromImage />
      <MenuDivider />
      <ImageMenuItemChangeBoard />
      <ImageMenuItemStarUnstar />
      <MenuDivider />
      <ImageMenuItemDelete />
    </ImageDTOContextProvider>
  );
};

export default memo(SingleSelectionMenuItems);
