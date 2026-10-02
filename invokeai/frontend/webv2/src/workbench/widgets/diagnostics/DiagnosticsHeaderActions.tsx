import { Badge, VisuallyHidden } from '@chakra-ui/react';
import { useProblemCount } from '@workbench/diagnostics/useProblemCount';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { BugIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const DiagnosticsHeaderActions = () => {
  const { t } = useTranslation();
  const problemCount = useProblemCount(useActiveProjectId());

  if (problemCount === 0) {
    return null;
  }

  return (
    <Badge colorPalette="red" size="xs">
      <BugIcon />
      <span aria-hidden>{problemCount}</span>
      <VisuallyHidden>{t('widgets.diagnostics.chipProblems', { count: problemCount })}</VisuallyHidden>
    </Badge>
  );
};
