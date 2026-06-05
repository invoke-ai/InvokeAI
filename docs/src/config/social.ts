import type { StarlightUserConfig } from '@astrojs/starlight/types';

type SocialConfig = StarlightUserConfig['social'];

const social: SocialConfig = [
  {
    icon: 'github',
    label: 'GitHub',
    href: 'https://github.com/invoke-ai/InvokeAI',
  },
  {
    icon: 'discord',
    label: 'Discord',
    href: 'https://discord.gg/ZmtBAhwWhy',
  },
  {
    icon: 'youtube',
    label: 'YouTube',
    href: 'https://www.youtube.com/@invokeai',
  },
];

export { social as socialConfig };
