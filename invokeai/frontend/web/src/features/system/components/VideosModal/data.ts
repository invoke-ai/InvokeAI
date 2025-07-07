/**
 * To add a support video, you'll need to add the video to the list below.
 *
 * The `tKey` is a sub-key in the translation file `invokeai/frontend/web/public/locales/en.json`.
 * Add the title and description under `supportVideos.videos`, following the existing format.
 */

export type VideoData = {
  tKey: string;
  link: string;
};

export const supportVideos: VideoData[] = [
  {
    tKey: 'gettingStarted',
    link: 'https://www.youtube.com/watch?v=jVi2XgSGrfY&list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO&pp=gAQB0gcJCV8EOCosWNin',
  },
  {
    tKey: 'studioSessions',
    link: 'https://www.youtube.com/watch?v=91ZgeeqL7Bo&list=PLvWK1Kc8iXGq_8tWZqnwDVaf9uhlDC09U&pp=gAQB',
  },
];

export const gettingStartedPlaylistLink = 'https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO';
export const studioSessionsPlaylistLink = 'https://www.youtube.com/playlist?list=PLvWK1Kc8iXGq_8tWZqnwDVaf9uhlDC09U';
