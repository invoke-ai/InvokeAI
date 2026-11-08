// The docs' icons, from Lucide like the app's. Where the app shows the same concept, the docs use the same icon (the
// app source is noted). Section icons are keyed by the section's directory as a URL path (the form `alsoListedIn`
// uses, e.g. `users-guide/workbench` for `Users Guide/01.Workbench`), so renumbering a directory keeps its icon. The
// build fails when a topic or sidebar section has no icon here.

import AppWindow from '@lucide/astro/icons/app-window';
import BookOpen from '@lucide/astro/icons/book-open';
import BookOpenText from '@lucide/astro/icons/book-open-text';
import Box from '@lucide/astro/icons/box';
import Brush from '@lucide/astro/icons/brush';
import Blocks from '@lucide/astro/icons/blocks';
import Clapperboard from '@lucide/astro/icons/clapperboard';
import Cloud from '@lucide/astro/icons/cloud';
import CodeXml from '@lucide/astro/icons/code-xml';
import Container from '@lucide/astro/icons/container';
import Cpu from '@lucide/astro/icons/cpu';
import FileText from '@lucide/astro/icons/file-text';
import FlaskConical from '@lucide/astro/icons/flask-conical';
import Folder from '@lucide/astro/icons/folder';
import Gauge from '@lucide/astro/icons/gauge';
import GitBranch from '@lucide/astro/icons/git-branch';
import GitPullRequest from '@lucide/astro/icons/git-pull-request';
import ImageIcon from '@lucide/astro/icons/image';
import Layers from '@lucide/astro/icons/layers';
import LifeBuoy from '@lucide/astro/icons/life-buoy';
import MapIcon from '@lucide/astro/icons/map';
import PanelsTopLeft from '@lucide/astro/icons/panels-top-left';
import Puzzle from '@lucide/astro/icons/puzzle';
import Rocket from '@lucide/astro/icons/rocket';
import Settings from '@lucide/astro/icons/settings';
import SlidersHorizontal from '@lucide/astro/icons/sliders-horizontal';
import Tag from '@lucide/astro/icons/tag';
import Users from '@lucide/astro/icons/users';
import Workflow from '@lucide/astro/icons/workflow';
import Wrench from '@lucide/astro/icons/wrench';

import { findSectionIcon } from '../lib/section-icon.mjs';
import { docsTopics, type DocsTopic } from './topics';

export type IconComponent = typeof Folder;

const topicIcons: Record<DocsTopic['id'], IconComponent> = {
  'get-started': Rocket,
  guide: BookOpen,
  models: Box, // Launchpad: Models
  configure: Settings, // Launchpad: Preferences
  developers: CodeXml,
  contribute: GitPullRequest,
  troubleshooting: LifeBuoy,
  releases: Tag,
};

const sectionIcons: Record<string, IconComponent> = {
  'users-guide/workbench': PanelsTopLeft, // What's new: the interface
  'users-guide/projects': Folder, // Launchpad: Projects
  'users-guide/image-generation': SlidersHorizontal, // Generate widget
  'users-guide/image-generation/advanced-topics': FlaskConical,
  'users-guide/video-generation': Clapperboard, // Video widget, launchpad video intent
  'users-guide/gallery': ImageIcon, // Gallery widget
  'users-guide/canvas': Brush, // Launchpad canvas intent
  'users-guide/workflows': Workflow, // Workflow widget
  'users-guide/image-map': MapIcon, // Image Map widget
  'users-guide/models/local-models': Cpu,
  'users-guide/models/external-models': Cloud, // External provider keys
  'configuration/optimization': Gauge,
  'configuration/multi-user-mode': Users, // Launchpad: Users
  'development/architecture': Layers,
  'development/documentation': FileText,
  'development/front-end': AppWindow, // Open projects control
  'development/guides': BookOpenText, // Help menu: documentation
  'development/process': GitBranch,
  'development/setup': Wrench,
};

/** The few pages that need an icon of their own, apart from their section's or topic's, keyed by page path. */
const pageIcons: Record<string, IconComponent> = {
  'configuration/docker': Container,
  'development/guides/creating-nodes': Puzzle,
  'development/guides/creating-node-pack': Blocks, // Node pack library and detail
};

/** A topic's icon. */
export const getTopicIcon = (topic: DocsTopic): IconComponent => {
  const icon = topicIcons[topic.id];
  if (!icon) throw new Error(`The docs topic "${topic.id}" has no icon. Add one to src/config/icons.ts.`);
  return icon;
};

/** A sidebar section's icon, by the section's directory as a URL path. */
export const getSidebarSectionIcon = (directory: string, label: string): IconComponent => {
  const icon = sectionIcons[directory];
  if (!icon) throw new Error(`The sidebar section "${label}" (${directory}) has no icon. Add one to src/config/icons.ts.`);
  return icon;
};

/**
 * A docs page's icon: its own if it has one, else its section's, else its topic's; `undefined` outside every topic.
 * @param path the page's path without the deployment base path, e.g. `/configuration/docker/`
 */
export const getSectionIcon = (path: string): IconComponent | undefined =>
  findSectionIcon(
    path,
    pageIcons,
    sectionIcons,
    docsTopics.map((topic) => ({ directory: topic.directory, icon: getTopicIcon(topic) }))
  );
