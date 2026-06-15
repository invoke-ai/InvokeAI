import { Badge } from '@chakra-ui/react';
import { BugIcon } from 'lucide-react';

import { Button } from '@workbench/components/ui/Button';
import { useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';

export const DiagnosticsHeaderActions = () => {
  const errorCount = useWorkbenchSelector((snapshot) => snapshot.state.errorLog.length);
  const dispatch = useWorkbenchDispatch();

  return (
    <>
      {errorCount ? (
        <Badge colorPalette="red" size="xs">
          <BugIcon />
          {errorCount}
        </Badge>
      ) : null}
      <Button
        disabled={errorCount === 0}
        size="2xs"
        variant="outline"
        onClick={() => dispatch({ type: 'clearErrorLog' })}
      >
        Clear
      </Button>
    </>
  );
};
