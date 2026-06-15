import { Badge } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { BugIcon } from 'lucide-react';

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
