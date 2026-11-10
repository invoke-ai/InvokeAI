import { readdirSync } from 'node:fs';

import type { StarlightUserConfig } from '@astrojs/starlight/types';
import { makeChangelogsSidebarLinks } from 'starlight-changelogs';
import type starlightSidebarTopics from 'starlight-sidebar-topics';

import { getSortPrefix, stripSortPrefix } from '../lib/sort-prefix.mjs';
import { docsTopics, type DocsTopic } from './topics';

type SidebarItems = NonNullable<StarlightUserConfig['sidebar']>;
type SidebarTopicsConfig = Parameters<typeof starlightSidebarTopics>[0];
type SidebarTopicsOptions = Parameters<typeof starlightSidebarTopics>[1];

const usersGuideDirectory = 'Users Guide';
const modelsSection = '03.Models';

/**
 * The Guide topic is the Users Guide without its Models section, which is a topic of its own. Each remaining section
 * becomes a group, so a new section joins the Guide without a config change.
 */
const guideItems: SidebarItems = readdirSync(new URL(`../content/docs/${usersGuideDirectory}/`, import.meta.url), {
  withFileTypes: true,
})
  .filter((entry) => entry.isDirectory() && entry.name !== 'assets' && entry.name !== modelsSection)
  .map((entry) => entry.name)
  .sort((a, b) => (getSortPrefix(a) ?? Infinity) - (getSortPrefix(b) ?? Infinity) || a.localeCompare(b))
  .map((name) => ({
    label: stripSortPrefix(name),
    collapsed: true,
    items: [{ autogenerate: { directory: `${usersGuideDirectory}/${name}` } }],
  }));

const topicItems: Record<DocsTopic['id'], SidebarItems> = {
  'get-started': [{ autogenerate: { directory: 'start-here' } }],
  guide: guideItems,
  models: [{ autogenerate: { directory: `${usersGuideDirectory}/${modelsSection}` } }],
  configure: [{ autogenerate: { directory: 'configuration' } }],
  developers: [{ autogenerate: { directory: 'development' } }],
  contribute: [{ autogenerate: { directory: 'contributing' } }],
  troubleshooting: [{ autogenerate: { directory: 'troubleshooting' } }],
  releases: makeChangelogsSidebarLinks([
    { type: 'all', label: 'All releases', base: 'releases' },
    { type: 'recent', base: 'releases' },
  ]),
};

const sidebarTopics: SidebarTopicsConfig = docsTopics.map(({ id, label, link }) => {
  const items = topicItems[id];
  if (!items) throw new Error(`The docs topic "${id}" has no sidebar items.`);
  return { id, label, link, items };
});

const sidebarTopicsOptions: SidebarTopicsOptions = {
  // Release pages beyond the recent ones the Releases sidebar links to.
  topics: { releases: ['/releases/**'] },
};

export { sidebarTopics, sidebarTopicsOptions };
