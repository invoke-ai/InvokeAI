import { Badge } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { clearProjectDiagnostics, useProjectDiagnostics } from '@workbench/diagnostics/logger';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { BugIcon } from 'lucide-react';
import { useCallback } from 'react';

export const DiagnosticsHeaderActions = () => {
  const projectId = useActiveProjectSelector((project) => project.id);
  const entries = useProjectDiagnostics(projectId);
  const clearEntries = useCallback(() => clearProjectDiagnostics(projectId), [projectId]);

  return (
    <>
      {entries.length ? (
        <Badge colorPalette="red" size="xs">
          <BugIcon />
          {entries.length}
        </Badge>
      ) : null}
      <Button disabled={entries.length === 0} size="2xs" variant="outline" onClick={clearEntries}>
        Clear
      </Button>
    </>
  );
};
