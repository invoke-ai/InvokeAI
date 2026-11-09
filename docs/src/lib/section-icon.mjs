/**
 * A docs page's icon: its own, if it has one, else that of the section it belongs to, else of its topic, else
 * `undefined`. Sections and topics match by the deepest of the page's directories that has an icon, a section's
 * icon winning over a topic's for the same directory.
 * @template Icon
 * @param {string} path the page's path without the deployment base path, e.g. `/configuration/docker/`
 * @param {Record<string, Icon>} pageIcons page path (as in a URL, without slashes at the ends) -> icon
 * @param {Record<string, Icon>} sectionIcons section directory (as a URL path) -> icon
 * @param {{ directory: string, icon: Icon }[]} topicIcons
 * @returns {Icon | undefined}
 */
export const findSectionIcon = (path, pageIcons, sectionIcons, topicIcons) => {
  const page = path.replace(/^\/+|\/+$/g, '');
  if (pageIcons[page]) return pageIcons[page];
  const segments = page.split('/');
  for (let length = segments.length; length > 0; length--) {
    const directory = segments.slice(0, length).join('/');
    const icon = sectionIcons[directory] ?? topicIcons.find((topic) => topic.directory === directory)?.icon;
    if (icon) return icon;
  }
  return undefined;
};
