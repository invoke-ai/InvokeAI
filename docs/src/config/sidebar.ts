import type { StarlightUserConfig } from '@astrojs/starlight/types';
import { makeChangelogsSidebarLinks } from 'starlight-changelogs';

type SidebarConfig = StarlightUserConfig['sidebar'];

const sidebar: SidebarConfig = [
  {
    label: 'Start Here',
    items: [
      {
        autogenerate: { directory: 'start-here' },
      },
    ],
  },
  {
    label: 'Configuration',
    items: [
      {
        autogenerate: { directory: 'configuration' },
      },
    ],
  },
  {
    label: 'Concepts',
    items: [
      {
        autogenerate: { directory: 'concepts' },
      },
    ],
  },
  {
    label: 'Features',
    items: [
      {
        autogenerate: { directory: 'features' },
      },
    ],
  },
  {
    label: 'Workflows',
    items: [
      {
        autogenerate: { directory: 'workflows' },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Development',
    items: [
      {
        autogenerate: { directory: 'development', collapsed: true },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Contributing',
    items: [
      {
        autogenerate: { directory: 'contributing' },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Troubleshooting',
    items: [
      {
        autogenerate: { directory: 'troubleshooting' },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Releases',
    collapsed: true,
    items: [
      ...makeChangelogsSidebarLinks([
        {
          type: 'recent',
          base: 'releases',
        },
      ]),
    ],
  },
];

export { sidebar as sidebarConfig };
