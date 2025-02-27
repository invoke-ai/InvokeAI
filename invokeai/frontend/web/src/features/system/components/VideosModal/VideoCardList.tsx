import { Divider } from '@invoke-ai/ui-library';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import { gettingStartedVideos, type VideoData } from 'features/system/components/VideosModal/data';
import { VideoCard } from 'features/system/components/VideosModal/VideoCard';
import { Fragment, memo } from 'react';

export const VideoCardList = memo(({ category, videos }: { category: string; videos: VideoData[] }) => {
  return (
    <StickyScrollable title={category}>
      {videos.map((video, i) => (
        <Fragment key={`${video.tKey}-${i}`}>
          <VideoCard video={video} />
          {i < gettingStartedVideos.length - 1 && <Divider />}
        </Fragment>
      ))}
    </StickyScrollable>
  );
});

VideoCardList.displayName = 'VideoCardList';
