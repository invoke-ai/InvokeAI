// The docs are split into topics, each with its own sidebar (see `sidebar.ts`). The topic switcher at the top of the
// sidebar lists them in these groups.

type TopicGroup = 'Use' | 'Run' | 'Build' | 'Help';

type DocsTopic = {
  /** Stable identifier, exposed to client scripts through the `docs-topic` meta tag. */
  id: string;
  /** Unique label, shown in the switcher and used as the Pagefind `topic` filter value. */
  label: string;
  description: string;
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
    description: 'Install, requirements, upgrades',
    group: 'Use',
    link: '/start-here/installation/',
    directory: 'start-here',
  },
  {
    id: 'guide',
    label: 'Guide',
    description: 'Workbench, canvas, gallery, workflows',
    group: 'Use',
    link: '/users-guide/workbench/introduction/',
    directory: 'users-guide',
  },
  {
    id: 'models',
    label: 'Models',
    description: 'Families, installs, providers',
    group: 'Use',
    link: '/users-guide/models/introduction/',
    directory: 'users-guide/models',
  },
  {
    id: 'configure',
    label: 'Configure & Deploy',
    description: 'invokeai.yaml, Docker, multi-user',
    group: 'Run',
    link: '/configuration/invokeai-yaml/',
    directory: 'configuration',
  },
  {
    id: 'developers',
    label: 'Developers',
    description: 'Architecture, nodes, APIs',
    group: 'Build',
    link: '/development/',
    directory: 'development',
  },
  {
    id: 'contribute',
    label: 'Contribute',
    description: 'Code, translations, process',
    group: 'Build',
    link: '/contributing/',
    directory: 'contributing',
  },
  {
    id: 'troubleshooting',
    label: 'Troubleshooting',
    description: 'FAQ, known bugs, videos',
    group: 'Help',
    link: '/troubleshooting/faq/',
    directory: 'troubleshooting',
  },
  {
    id: 'releases',
    label: 'Releases',
    description: 'Changelog for every version',
    group: 'Help',
    link: '/releases/',
    directory: 'releases',
  },
];

/** The topic of the page being rendered, or `undefined` for pages outside every topic (home, download, 404). */
const getCurrentTopic = (locals: App.Locals): DocsTopic | undefined => {
  const current = locals.starlightSidebarTopics?.topics.find((topic) => topic.isCurrent);
  return current && docsTopics.find((topic) => topic.label === current.label);
};

export { docsTopics, getCurrentTopic, topicGroups };
export type { DocsTopic, TopicGroup };
