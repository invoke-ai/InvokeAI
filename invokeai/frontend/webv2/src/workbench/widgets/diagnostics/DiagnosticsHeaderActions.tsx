import { Badge } from '@chakra-ui/react';
import { BugIcon } from 'lucide-react';

import { Button } from '../../components/ui/Button';
import { useWorkbench } from '../../WorkbenchContext';

export const DiagnosticsHeaderActions = () => {
  const { dispatch, state } = useWorkbench();

  return (
    <>
      {state.errorLog.length ? (
        <Badge colorPalette="red" size="xs">
          <BugIcon />
          {state.errorLog.length}
        </Badge>
      ) : null}
      <Button
        disabled={state.errorLog.length === 0}
        size="2xs"
        variant="outline"
        onClick={() => dispatch({ type: 'clearErrorLog' })}
      >
        Clear
      </Button>
    </>
  );
};
