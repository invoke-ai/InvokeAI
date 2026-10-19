/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { ListContextMenuAnchor } from '@platform/ui/list/ListItem';
import type { ReactNode } from 'react';

import { ListItem } from '@platform/ui/list/ListItem';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { memo } from 'react';

import type { InstallSourceStatus } from './InstallSourceMenu';

import { InstallSourceButton } from './InstallSourceButton';
import { sourceFileName } from './useSourceNameFilter';

interface InstallSourceRowProps extends InstallSourceStatus {
  /** This row is the target of its list's context menu. */
  isMenuOpen: boolean;
  onOpenMenu: (source: string, anchor: ListContextMenuAnchor) => void;
  /** Source string used to match an active install job. */
  source: string;
}

/**
 * An installable source and its Install button. Right-click opens the list's shared menu for this source; the row
 * itself is not a tab stop, since the button already offers the same action to the keyboard.
 */
export const InstallSourceRow = ({
  badges,
  description,
  isMenuOpen,
  name,
  onInstall,
  onOpenMenu,
  source,
  title,
  ...status
}: InstallSourceRowProps & {
  badges?: ReactNode;
  description?: ReactNode;
  /** What installs, so repeated Install controls stay distinguishable to assistive tech. */
  name: string;
  onInstall: () => void;
  title: string;
}) => (
  <ListItem
    actions={<InstallSourceButton {...status} name={name} source={source} onInstall={onInstall} />}
    badges={badges}
    description={description}
    isMenuOpen={isMenuOpen}
    tabIndex={-1}
    title={title}
    onContextMenu={(anchor) => onOpenMenu(source, anchor)}
  />
);

/**
 * A file source (a scanned path or a repository URL), memoized so opening the list's menu re-renders only the
 * targeted row; callers pass stable callbacks.
 */
export const InstallPathRow = memo(function InstallPathRow({
  location,
  onInstall,
  ...row
}: InstallSourceRowProps & {
  /** Where the file sits under its scan root or repository, shown under the file name. */
  location: string | null;
  onInstall: (source: string) => void;
}) {
  const fileName = sourceFileName(row.source);

  return (
    <InstallSourceRow
      {...row}
      description={location ? <MiddleTruncate as="span" text={location} title={row.source} /> : undefined}
      name={location ?? fileName}
      title={fileName}
      onInstall={() => onInstall(row.source)}
    />
  );
});
