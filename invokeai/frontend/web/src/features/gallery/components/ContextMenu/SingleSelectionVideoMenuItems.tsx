import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { ContextMenuItemChangeBoard } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemChangeBoard';
import { ContextMenuItemDownload } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDownload';
import { ContextMenuItemOpenInNewTab } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInNewTab';
import { ContextMenuItemOpenInViewer } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInViewer';
import { ItemDTOContextProvider } from 'features/gallery/contexts/ItemDTOContext';
import type { VideoDTO } from 'services/api/types';

import { ContextMenuItemDeleteVideo } from './MenuItems/ContextMenuItemDeleteVideo';
import { ContextMenuItemStarUnstar } from './MenuItems/ContextMenuItemStarUnstar';

type SingleSelectionVideoMenuItemsProps = {
  videoDTO: VideoDTO;
};

const SingleSelectionVideoMenuItems = ({ videoDTO }: SingleSelectionVideoMenuItemsProps) => {
  return (
    <ItemDTOContextProvider value={videoDTO}>
      <IconMenuItemGroup>
        <ContextMenuItemOpenInNewTab />
        <ContextMenuItemDownload />
        <ContextMenuItemOpenInViewer />
        <ContextMenuItemDeleteVideo />
      </IconMenuItemGroup>
      <MenuDivider />
      <ContextMenuItemStarUnstar />
      <ContextMenuItemChangeBoard />
    </ItemDTOContextProvider>
  );
};

export default SingleSelectionVideoMenuItems;
