import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box } from '@chakra-ui/react';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { settingsCatalog } from './catalog';
import { SettingsEntryView } from './SettingsEntryView';

const QUICK_GROUP_STYLES = { '& > h3:first-of-type': { paddingTop: '0' } };

const WidgetQuickSettings = ({ manifest, instance }: WidgetViewProps) => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const target = useMemo(() => ({ projectId, instanceId: instance.id }), [projectId, instance.id]);
  const section = settingsCatalog.find((candidate) => candidate.id === manifest.settings?.id);
  if (!section) {
    return null;
  }
  const entries =
    manifest.settings?.quick?.flatMap((id) => section.entries.filter((entry) => entry.field.id === id)) ?? [];
  return (
    <Box css={QUICK_GROUP_STYLES}>
      {entries.map((entry, index) => {
        const group = entry.field.group ? resolveSettingsText(entry.field.group, t) : '';
        const previousGroup = entries[index - 1]?.field.group;
        const showGroup = Boolean(group && group !== (previousGroup ? resolveSettingsText(previousGroup, t) : ''));
        return (
          <SettingsEntryView
            key={entry.field.id}
            entry={entry}
            section={section}
            target={target}
            surface="quick"
            showGroup={showGroup}
          />
        );
      })}
    </Box>
  );
};
export default WidgetQuickSettings;
