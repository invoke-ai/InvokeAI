// The docs are split into topics, each with its own sidebar (see `sidebar.ts`). The topic switcher at the top of the
// sidebar lists them in these groups. Their displayed names and descriptions are UI strings (`topics.*` keys in
// `src/content/i18n/en.json`).

type TopicGroup = 'Use' | 'Run' | 'Build' | 'Help';

type DocsTopic = {
  /** Stable identifier: the Pagefind `topic` filter value, also exposed through the `docs-topic` meta tag. */
  id: string;
  /**
   * Unique English label, by which starlight-sidebar-topics identifies the current topic. Pages show `topicLabel`.
   */
  label: string;
  group: TopicGroup;
  /** The topic's landing page, without the deployment base path. */
  link: string;
  /** The URL path its pages live under, for icon lookups (see `icons.ts`). */
  directory: string;
};

const topicGroups: TopicGroup[] = ['Use', 'Run', 'Build', 'Help'];

const docsTopics: DocsTopic[] = [
  {
    id: 'get-started',
    label: 'Get Started',
    group: 'Use',
    link: '/start-here/installation/',
    directory: 'start-here',
  },
  {
    id: 'guide',
    label: 'Guide',
    group: 'Use',
    link: '/users-guide/workbench/introduction/',
    directory: 'users-guide',
  },
  {
    id: 'models',
    label: 'Models',
    group: 'Use',
    link: '/users-guide/models/introduction/',
    directory: 'users-guide/models',
  },
  {
    id: 'configure',
    label: 'Configure & Deploy',
    group: 'Run',
    link: '/configuration/invokeai-yaml/',
    directory: 'configuration',
  },
  {
    id: 'developers',
    label: 'Developers',
    group: 'Build',
    link: '/development/',
    directory: 'development',
  },
  {
    id: 'contribute',
    label: 'Contribute',
    group: 'Build',
    link: '/contributing/',
    directory: 'contributing',
  },
  {
    id: 'troubleshooting',
    label: 'Troubleshooting',
    group: 'Help',
    link: '/troubleshooting/faq/',
    directory: 'troubleshooting',
  },
  {
    id: 'releases',
    label: 'Releases',
    group: 'Help',
    link: '/releases/',
    directory: 'releases',
  },
];

type Translate = App.Locals['t'];

const topicLabel = (t: Translate, topic: DocsTopic) => t(`topics.${topic.id}.label`);
const topicDescription = (t: Translate, topic: DocsTopic) => t(`topics.${topic.id}.description`);
const topicGroupLabel = (t: Translate, group: TopicGroup) => t(`topics.group.${group.toLowerCase()}`);

/** The topic of the page being rendered, or `undefined` for pages outside every topic (home, download, 404). */
const getCurrentTopic = (locals: App.Locals): DocsTopic | undefined => {
  const current = locals.starlightSidebarTopics?.topics.find((topic) => topic.isCurrent);
  return current && docsTopics.find((topic) => topic.label === current.label);
};

export { docsTopics, getCurrentTopic, topicDescription, topicGroupLabel, topicGroups, topicLabel };
export type { DocsTopic, TopicGroup };
