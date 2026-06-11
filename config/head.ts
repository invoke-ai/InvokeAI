import type { StarlightUserConfig } from '@astrojs/starlight/types';

type HeadConfig = NonNullable<StarlightUserConfig['head']>;

type CreateHeadConfigParams = {
  base: string;
  enableAnalytics: boolean;
  isGhPages: boolean;
  site: string;
};

const plausibleScriptUrl =
  'https://plausible.tracking.events/js/pa-BHcumuOemKz4XIQeWkTn4.js';
const plausibleInitScript =
  'window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};plausible.init()';

function createHeadConfig({
  base,
  enableAnalytics,
  isGhPages,
  site,
}: CreateHeadConfigParams): HeadConfig {
  const coverImageUrl = new URL(`${base}/coverimage.png`, site).toString();

  return [
    {
      tag: 'meta',
      attrs: {
        property: 'og:image',
        content: coverImageUrl,
      },
    },
    {
      tag: 'meta',
      attrs: {
        property: 'og:image:width',
        content: '1200',
      },
    },
    {
      tag: 'meta',
      attrs: {
        property: 'og:image:height',
        content: '630',
      },
    },
    {
      tag: 'meta',
      attrs: {
        name: 'twitter:card',
        content: 'summary_large_image',
      },
    },
    {
      tag: 'meta',
      attrs: {
        name: 'twitter:image',
        content: coverImageUrl,
      },
    },
    ...(enableAnalytics && !isGhPages
      ? ([
          {
            tag: 'script',
            attrs: {
              async: true,
              src: plausibleScriptUrl,
            },
          },
          {
            tag: 'script',
            content: plausibleInitScript,
          },
        ] satisfies HeadConfig)
      : []),
  ] satisfies HeadConfig;
}

export { createHeadConfig };
