import { Divider } from '@invoke-ai/ui-library';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { PlaylistData } from 'features/system/components/VideosModal/data';
import { PlaylistCard } from 'features/system/components/VideosModal/PlaylistCard';
import { Fragment, memo } from 'react';

export const PlaylistCardList = memo(({ category, playlists }: { category: string; playlists: PlaylistData[] }) => {
  return (
    <StickyScrollable title={category}>
      {playlists.map((playlist, i) => (
        <Fragment key={`${playlist.tKey}-${i}`}>
          <PlaylistCard playlist={playlist} />
          {i < playlists.length - 1 && <Divider />}
        </Fragment>
      ))}
    </StickyScrollable>
  );
});

PlaylistCardList.displayName = 'PlaylistCardList';
